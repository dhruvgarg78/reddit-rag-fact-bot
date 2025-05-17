# 🧠 Reddit Historical Fact Checker (RAG-Based)

This project is a Retrieval-Augmented Generation (RAG) pipeline that monitors Reddit posts related to Indian history and replies with fact-checked information based on **scholarly sources**. It integrates document embedding, FAISS-based retrieval, and Gemini-based language generation.

---

## 📁 Project Structure

### `main.py`
Core logic for processing documents and building the RAG pipeline:
- 📄 **Text Extraction**: Uses OCR (via `ocrmypdf` + `PyMuPDF`) to convert historical PDFs into raw text.
- ✂️ **Chunking**: Breaks text into overlapping segments with metadata.
- 📌 **Embedding**: Converts chunks into dense vectors using `sentence-transformers`.
- 📂 **Indexing**: Stores embeddings in a FAISS index for efficient retrieval.
- 🔍 **Querying**: Retrieves relevant chunks using semantic search.
- 🧩 **Prompt Creation**: Constructs a context-aware prompt from retrieved chunks.
- 🤖 **Gemini Interaction**: Queries Google's Gemini model with the prompt.
- 🧠 **Query Variants**: Rewrites the input query into multiple semantically related versions to improve recall.

> Includes an optional one-time `build_index_from_pdfs()` function to generate the FAISS index.

### `reddit_fact_checker.py`
A Reddit bot that:
- Uses the `praw` library to connect to Reddit.
- Scans `r/factcheckbot_testing` for posts mentioning keywords (e.g., "Mughal", "Shivaji", "Panipat").
- For relevant posts:
  - Generates query variants.
  - Retrieves related chunks using FAISS.
  - Builds a context-rich Gemini prompt.
  - Posts the Gemini-generated fact-check as a reply.

---

## 📚 Historical Sources Used

This project grounds its fact-checking in the following **primary scholarly texts**, as defined in `main.py`:

1. **British History (1782–1919)**  
   *By George Macaulay Trevelyan*  
   📄 [Archive.org link](https://archive.org/details/in.ernet.dli.2015.228096/page/n5/mode/2up)

2. **Maratha History**  
   *By G.S. Sardesai, Patwardhan & Rawlinson*  
   📄 [Archive.org link](https://archive.org/details/in.ernet.dli.2015.514342)

3. **Mughal History**  
   *Various Sources: J. N. Chaudhuri, Journal of the Pakistan Historical Society (JPHS), Indian Historical Records Commission*  
   📄 [Archive.org link](https://archive.org/details/mughal-empire-r.-c.-majumdar-1974)

These documents cover key events, figures, and political dynamics from 17th–20th century India.

---

## 🔧 Setup Instructions

1. **Clone Repository** and place the 3 PDFs in the root directory.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Create `.env`** file with:
   ```env
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   REDDIT_USERNAME=...
   REDDIT_PASSWORD=...
   REDDIT_USER_AGENT=...
   GEMINI_API_KEY=...
   ```
4. **Build FAISS Index** (run once):
   ```python
   # In main.py
   build_index_from_pdfs()
   ```
5. **Run Reddit Bot**:
   ```bash
   python reddit_fact_checker.py
   ```

---

## 💡 Example Use Case

If a Reddit post says:  
> “In Chhaava, they show Sambhaji being humiliated before execution by Aurangzeb.”

The bot will:
- Detect it contains historical keywords.
- Generate variants of this query.
- Retrieve relevant historical context chunks.
- Construct a Gemini prompt with that context.
- Post a historically-grounded fact check reply citing the exact sources.

---

## ✅ TODOs
- [ ] Add unit tests
- [ ] Improve query deduplication logic
- [ ] Extend support to other subreddits
- [ ] Fine-tune chunking and retrieval parameters
