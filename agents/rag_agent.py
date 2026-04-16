import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents(docs_folder: str = "rag_docs") -> list:
    documents = []
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"📁 Created '{docs_folder}/' folder — add your HR .txt documents there!")
        return []

    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_folder, filename), "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"filename": filename, "content": content})
                print(f"   📄 Loaded: {filename}")
    return documents


def build_vector_store(documents: list):
    contents = [doc["content"] for doc in documents]
    embeddings = embedding_model.encode(contents)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, contents


def retrieve_relevant_context(query: str, index, contents: list, top_k: int = 3) -> str:
    query_embedding = embedding_model.encode([query]).astype("float32")
    top_k = min(top_k, len(contents))
    distances, indices = index.search(query_embedding, top_k)

    results = [contents[i] for i in indices[0] if i < len(contents)]
    if not results:
        return "No relevant documents found."

    context = "\n\n---\n\n".join(results)
    return context


def run_rag_agent(eda_results: dict) -> dict:
    print("🤖 Agent 6 [RAG] — Starting...")

    print("   📂 Loading HR documents from rag_docs/...")
    documents = load_documents("rag_docs")

    if not documents:
        print("   ⚠️ No documents found in rag_docs/ folder.")
        print("   ℹ️ Add .txt HR documents to rag_docs/ for richer insights.")
        print("   ✅ Agent 6 Done! (No RAG context — using data only)")
        return {
            "rag_context": "",
            "eda_results": eda_results
        }

    print("   🔢 Building vector store...")
    index, contents = build_vector_store(documents)

    print("   🔍 Retrieving relevant HR context...")
    query = f"""
    HR attrition analysis. Overall attrition {eda_results['attrition_rate']}%.
    Sales department attrition highest. Overtime strongly linked to attrition.
    Average age of employees who left is {eda_results['avg_age_left']}.
    """
    context = retrieve_relevant_context(query, index, contents)

    print("   ✅ Agent 6 Done! Relevant context retrieved.")
    print("\n--- RAG CONTEXT PREVIEW ---")
    print(context[:300] + "...")

    return {
        "rag_context": context,
        "eda_results": eda_results
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingestion_agent import run_ingestion_agent
    from eda_agent import run_eda_agent

    data = run_ingestion_agent("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    eda = run_eda_agent(data)
    run_rag_agent(eda)
