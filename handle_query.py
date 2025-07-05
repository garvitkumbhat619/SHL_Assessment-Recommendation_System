import os
import json
import numpy as np
import trafilatura
import faiss
from InstructorEmbedding import INSTRUCTOR
import torch
def extract_text_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else ""
def load_index_and_metadata(index_path, texts_path, metadata_path):
    if not all(map(os.path.exists, [index_path, texts_path, metadata_path])):
        raise FileNotFoundError("Required files not found in 'outputs/' folder.")
    index = faiss.read_index(index_path)
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, texts, metadata
def embed_query_instructor(query, device="cpu"):
    model = INSTRUCTOR("hkunlp/instructor-xl")
    model.to(device)
    instruction = "Represent the task: retrieve relevant assessments based on this job description"
    embedding = model.encode([[instruction, query]], convert_to_numpy=True, device=device)
    faiss.normalize_L2(embedding)
    return embedding
def parse_duration(duration_str):
    if not duration_str:
        return None
    try:
        digits = ''.join(filter(str.isdigit, str(duration_str)))
        return int(digits) if digits else None
    except:
        return None
def search_similar_fuzzy(query_vector, index, metadata, top_k=10, max_duration=None, required_types=None, type_penalty=0.8):
    D, I = index.search(query_vector, top_k * 5)
    candidates = []
    for idx, score in zip(I[0], D[0]):
        item = metadata[idx]
        name = item.get("Assessment Name", "UNKNOWN")
        url = item.get("URL", "N/A")
        duration = parse_duration(item.get("Duration", ""))
        std_type = item.get("Standardized Test Type", "").lower()  # using cleaned field
        if max_duration is not None and (duration is None or duration > max_duration):
            continue
        # Calculate raw similarity (lower FAISS L2 means higher similarity)
        similarity = 1 - score
        if required_types:
            if not any(req.lower() in std_type for req in required_types):
                similarity *= type_penalty
        candidates.append((name, url, similarity))
    # Sort candidates by adjusted similarity (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_k]
if __name__ == "__main__":
    print("=== SHL Assessment Recommender ===")
    user_input = input("Enter a job description or a URL: ").strip()
    if user_input.startswith("http"):
        print("Extracting content from URL...")
        text = extract_text_from_url(user_input)
        if not text:
            print("Failed to extract content from URL.")
            exit()
    else:
        text = user_input
    print("Embedding your query...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_vec = embed_query_instructor(text, device=device)
    print("Loading SHL assessment data...")
    # Make sure to load cleaned metadata here!
    index, texts, metadata = load_index_and_metadata(
        "outputs/faiss_index.idx",
        "outputs/assessment_texts.json",
        "shl_metadata_index_cleaned.json"
    )
    print("Finding top relevant assessments...")
    filtered_results = search_similar_fuzzy(
        query_vec,
        index,
        metadata,
        top_k=10,
        max_duration=45,
        required_types=["technical", "cognitive", "personality"]  # required types as needed
    )
    print("\nTop Matches:")
    for name, url, score in filtered_results:
        print(f"- {name} (Adjusted Similarity: {score:.4f})")
        print(f"  {url}\n")
