# reddit_fact_checker.py

import os
import time
import praw
from dotenv import load_dotenv
from main import (
    generate_query_variants_for_lookup,
    load_index_and_chunks,
    retrieve_top_k_chunks,
    build_prompt_from_chunks,
    query_gemini
)

# Load environment variables
load_dotenv()

# Connect to Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Keywords to search for
KEYWORDS = [
    "Mughal", "Maratha", "British", "Aurangzeb",
    "Shivaji", "Sambhaji", "Panipat", "colonial rule"
]

# Load FAISS index and chunks once
index, chunks = load_index_and_chunks()

# Main loop
def fact_check_reddit_posts():
    print("\n[Bot] Scanning new posts in r/factcheckbot_testing...\n")

    for submission in reddit.subreddit("factcheckbot_testing").new(limit=10):
        content = (submission.title + " " + submission.selftext).lower()

        if any(keyword.lower() in content for keyword in KEYWORDS):
            print(f"[Found] Relevant post: {submission.title}")

            # Skip if we've already replied
            if any(comment.author == reddit.user.me() for comment in submission.comments):
                print("↪️ Already replied. Skipping...\n")
                continue

            # Generate fact check
            query_variants = generate_query_variants_for_lookup(content)
            retrieved_chunks_with_queries = []
            for variant in query_variants:
                top_chunks = retrieve_top_k_chunks(variant, index, chunks, k=3)
                for chunk in top_chunks:
                    retrieved_chunks_with_queries.append({"chunk": chunk, "query": variant})

            # Deduplicate by (source, text) instead of chunk ID
            unique_chunks = {(c["chunk"]["source"], c["chunk"]["text"]): c for c in retrieved_chunks_with_queries}
            final_chunks = list(unique_chunks.values())

            prompt = build_prompt_from_chunks(content, final_chunks)

            answer = query_gemini(prompt)

            # Post the comment
            reply_text = (
                f"📜 **Fact Check:**\n\n{answer.strip()}\n\n---\n"
                f"^(I’m a bot trained on historical sources. Contact u/{os.getenv('REDDIT_USERNAME')} for feedback.)"
            )
            reply_text += (
                "\n\n**Sources:**\n"
                "- [British History (1782–1919), Trevelyan](https://archive.org/details/in.ernet.dli.2015.228096/page/n5/mode/2up)\n"
                "- [Maratha History, Patwardhan & Rawlinson](https://archive.org/details/in.ernet.dli.2015.514342)\n"
                "- [Mughal History, J. N. Chaudhuri et al.](https://archive.org/details/mughal-empire-r.-c.-majumdar-1974)"
            )


            submission.reply(reply_text)
            print("✅ Replied successfully!\n")
            time.sleep(10)  # Pause before next to be respectful of API

# Run continuously
if __name__ == "__main__":
    while True:
        try:
            fact_check_reddit_posts()
            print("[Sleeping] Waiting 10 minutes...\n")
            time.sleep(600)
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(60)
