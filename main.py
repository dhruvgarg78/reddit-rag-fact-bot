# -- Imports --
import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF
import ocrmypdf
import google.generativeai as genai

# -- Source Mapping --
SOURCE_LABEL_MAP = {
    "British_Hist.pdf": "British History (1782–1919, Trevelyan)",
    "Maratha_Hist.pdf": "Maratha History (Patwardhan & Rawlinson)",
    "Mughal_Hist.pdf": "Mughal History (Various Sources)"
}

# -- Step 1: Extract Text from PDF --
def extract_text_from_pdf(file_path, force_ocr=True):
    try:
        print(f"[Extracting] {file_path} ...")
        searchable_pdf = "searchable_temp.pdf"
        ocrmypdf.ocr(file_path, searchable_pdf, use_threads=True, force_ocr=force_ocr)
        doc = fitz.open(searchable_pdf)
        full_text = ""

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            print(f"--- Page {i+1} ---\n{text[:200]}{'...' if len(text) > 200 else ''}")
            full_text += text + "\n\n"

        doc.close()
        os.remove(searchable_pdf)
        print(f"[Done] Extracted {len(full_text)} characters from {file_path}")
        return full_text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {file_path}: {e}")
        return ""

# -- Step 2: Chunk with Metadata --
def chunk_with_metadata(text, source_name, chunk_size=300, overlap=50):
    print(f"[Chunking] {source_name} ...")
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[i:i + chunk_size])
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "chunk_id": i // (chunk_size - overlap)
            })

    print(f"[Done] Created {len(chunks)} chunks for {source_name}")
    return chunks

# -- Step 3: Embed Chunks --
def embed_chunks(chunks, model_name='intfloat/e5-small'):
    print(f"[Embedding] {len(chunks)} chunks using {model_name} ...")
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    print("[Done] Embedding complete.")
    return embeddings

# -- Step 4: Store in FAISS --
def store_in_faiss(embeddings):
    print("[Storing] Adding embeddings to FAISS index ...")
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    print(f"[Done] FAISS index now contains {index.ntotal} vectors.")
    return index

# -- Step 5: Save to Disk --
def save_index_and_chunks(index, chunks, index_path="rag_index.faiss", chunks_path="rag_chunks.pkl"):
    print("[Saving] Writing FAISS index and chunks to disk ...")
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[Done] Saved index to '{index_path}' and chunks to '{chunks_path}'")

# -- Step 6: Load from Disk --
def load_index_and_chunks(index_path="rag_index.faiss", chunks_path="rag_chunks.pkl"):
    print("[Loading] Loading FAISS index and chunks from disk ...")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"[Done] Loaded {len(chunks)} chunks.")
    return index, chunks

# -- Step 7: Retrieve Chunks --
def retrieve_top_k_chunks(query, index, chunks, model_name='intfloat/e5-small', k=3):
    print(f"\n🔍 [Query] {query}")
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding).astype('float32'), k)

    results = []
    for rank, (i, dist) in enumerate(zip(I[0], D[0])):
        chunk = chunks[i]
        print(f"\n--- Chunk #{rank+1} ---")
        print(f"📄 Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']}")
        print(f"📏 Distance: {dist:.4f}")
        print(f"📜 Preview: {chunk['text'][:200].replace(chr(10), ' ')}...")
        results.append(chunk)
    return results

# -- Step 8: Prompt Construction --
def build_prompt_from_chunks(query, retrieved_chunks_with_queries):
    """
    Builds a prompt for Gemini where each chunk includes which query it came from.
    This helps Gemini understand the context and relevance of each chunk.
    """

    context_sections = []
    for item in retrieved_chunks_with_queries:
        chunk = item["chunk"]
        variant = item["query"]
        doc_label = SOURCE_LABEL_MAP.get(chunk["source"], chunk["source"])
        context_sections.append(
            f"[Source: {doc_label} | Matched Query: \"{variant}\"]\n{chunk['text']}"
        )

    context = "\n\n".join(context_sections)

    prompt = (
        "You are a historical fact-checking assistant replying to a Reddit post.\n\n"
        "Your job is to respond as accurately as possible using ONLY the provided historical context.\n"
        "Do not hallucinate or include modern interpretations.\n\n"
        "You are referencing these historical documents:\n"
        "- British History (1782–1919, Trevelyan): https://archive.org/details/in.ernet.dli.2015.228096/page/n5/mode/2up\n"
        "- Maratha History (Patwardhan & Rawlinson): https://archive.org/details/in.ernet.dli.2015.514342\n"
        "- Mughal History (Various Sources: J. N. Chaudhuri, JPHS, Indian Historical Records Commission): https://archive.org/details/mughal-empire-r.-c.-majumdar-1974\n\n"
        "When citing, use the full document name, author and link. For example:\n"
        "Group citations at the end of the entire text you generate. Do not cite the same source repeatedly.\n\n"
        f"Reddit Post:\n{query}\n\n"
        f"Historical Context:\n{context}\n\n"
        "Fact-Check Reply:"
    )


    return prompt


# -- Step 9: Query Gemini --
def query_gemini(prompt, model_name="gemini-2.0-flash"):
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

# -- Generate Query Variants --
def generate_query_variants_for_lookup(original_query, model_name="gemini-2.0-flash"):
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)

    corpus_context = (
        "You are helping retrieve information from historical documents, including:\n"
        "- British History (1782–1919, Trevelyan)\n"
        "- Maratha History (by Patwardhan & Rawlinson)\n"
        "- Mughal Empire sources (various scholarly works)\n\n"
        "These documents cover historical events, people, battles, governance, and politics "
        "from the 17th to early 20th centuries. They do NOT include modern movies or people.\n"
    )

    instruction = (
        "Rewrite the user's query into 5 alternate versions for use in historical vector search. "
        "Avoid modern terms. Include one keyword-style variant.\n\n"
        f"{corpus_context}"
        f"Original query: {original_query}\n\n"
        "Return exactly 5 lines."
    )

    response = model.generate_content(instruction)
    variants = [line.strip() for line in response.text.split("\n") if line.strip()]
    print("🧠 Query variants generated:")
    for i, variant in enumerate(variants, 1):
        print(f"{i}. {variant}")
    return variants

# -- Step 10: Build Index (One-Time Use) --
def build_index_from_pdfs():
    pdf_files = [
        ("British_Hist.pdf", "British_Hist.pdf"),
        ("Maratha_Hist.pdf", "Maratha_Hist.pdf"),
        ("Mughal_Hist.pdf", "Mughal_Hist.pdf")
    ]

    all_chunks = []
    for path, label in pdf_files:
        text = extract_text_from_pdf(path)
        if not text:
            print(f"[ERROR] No text extracted from {path}. Skipping.")
            continue
        all_chunks.extend(chunk_with_metadata(text, label))

    embeddings = embed_chunks(all_chunks)
    index = store_in_faiss(embeddings)
    save_index_and_chunks(index, all_chunks)

# -- Main Execution --
if __name__ == "__main__":
    # Uncomment to build index initially
    # build_index_from_pdfs()

    index, chunks = load_index_and_chunks()

    query = """In Chhaava, they claim Aurangzeb executed Sambhaji Maharaj after a public humiliation — is this historically accurate?"""

    query_variants = generate_query_variants_for_lookup(query)

    retrieved_chunks = []
    for variant in query_variants:
        retrieved_chunks.extend(retrieve_top_k_chunks(variant, index, chunks, k=5))

    # Wrap each chunk with its corresponding query variant
    retrieved_chunks_with_queries = []
    for variant in query_variants:
        top_chunks = retrieve_top_k_chunks(variant, index, chunks, k=5)
        for chunk in top_chunks:
            retrieved_chunks_with_queries.append({"chunk": chunk, "query": variant})

    # Deduplicate based on (source, text) to avoid repeated chunks
    unique_wrapped = {(c["chunk"]["source"], c["chunk"]["text"]): c for c in retrieved_chunks_with_queries}
    final_wrapped_chunks = list(unique_wrapped.values())

    prompt = build_prompt_from_chunks(query, final_wrapped_chunks)

    answer = query_gemini(prompt)

    print("\n\nAnswer:\n", answer)
