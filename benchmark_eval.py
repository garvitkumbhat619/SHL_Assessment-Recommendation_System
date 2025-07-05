import json
import numpy as np
import pandas as pd
import faiss
from datetime import datetime
from InstructorEmbedding import INSTRUCTOR
import torch
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
def load_resources():
    index = faiss.read_index("outputs/faiss_index.idx")
    with open("outputs/assessment_texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open("shl_metadata_index_cleaned.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, texts, metadata
def load_eval_set(path="query_eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def parse_duration(duration_str):
    if not duration_str:
        return None
    try:
        digits = ''.join(filter(str.isdigit, str(duration_str)))
        return int(digits) if digits else None
    except:
        return None
def embed_query_instructor(queries, device="cpu"):
    model = INSTRUCTOR("hkunlp/instructor-xl")
    model.to(device)
    instruction = "Represent the task: retrieve relevant assessments based on this job description"
    pairs = [[instruction, q] for q in queries]
    vectors = model.encode(pairs, show_progress_bar=True, convert_to_numpy=True, device=device)
    faiss.normalize_L2(vectors)
    return vectors
def evaluate_map_recall(index, texts, metadata, eval_set, k=3, max_duration=None, required_types=None):
    recall_list = []
    map_list = []
    all_outputs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    queries = [q["query"] for q in eval_set]
    query_vecs = embed_query_instructor(queries, device=device)
    for i, entry in enumerate(eval_set):
        relevant_ids = set(entry["relevant_ids"])
        # Raw retrieval from FAISS (for debugging)
        D, I = index.search(np.array([query_vecs[i]]), k * 5)
        raw_retrieved = [metadata[idx].get("Assessment Name", "Unknown") for idx in I[0]]
        print("\n====================")
        print(f"Query: {entry['query']}")
        print(f"Expected (Relevant IDs): {relevant_ids}")
        print(f"Raw Retrieved (all candidates): {raw_retrieved}")
        filtered_results = []
        for idx, score in zip(I[0], D[0]):
            item = metadata[idx]
            name = item.get("Assessment Name", "Unknown")
            duration = parse_duration(item.get("Duration", ""))
            # Get the cleaned Test Type as a list (already standardized)
            std_types = [t.lower() for t in item.get("Test Type", [])]
            # Check duration: skip if duration is missing or exceeds max_duration.
            if max_duration is not None and (duration is None or duration > max_duration):
                print(f"Skipping {name} due to duration: {duration}")
                continue
            # Check required test types: Skip if none of the required types match.
            if required_types:
                if not any(req.lower() in std_types for req in required_types):
                    print(f"Skipping {name} due to test type mismatch: {std_types}")
                    continue
            adjusted_similarity = 1 - score  # No penalty here; candidate passes filter
            filtered_results.append((name, adjusted_similarity))
            if len(filtered_results) == k:
                break
        retrieved_names = [name for name, sim in filtered_results]
        print(f"Filtered Retrieved: {retrieved_names}")
        hits = [1 if name in relevant_ids else 0 for name, _ in filtered_results]
        recall = sum(hits) / len(relevant_ids) if relevant_ids else 0
        recall_list.append(recall)
        ap = 0.0
        num_hits = 0
        for j, hit in enumerate(hits):
            if hit:
                num_hits += 1
                ap += num_hits / (j + 1)
        ap = ap / min(k, len(relevant_ids)) if relevant_ids else 0
        map_list.append(ap)
        all_outputs.append({
            "Query": entry["query"],
            "Relevant Assessments": ", ".join(relevant_ids),
            "Retrieved Assessments": ", ".join([f"{name} ({sim:.4f})" for name, sim in filtered_results]),
            f"Recall@{k}": round(recall, 4),
            f"MAP@{k}": round(ap, 4)
        })
    metrics = {
        f"Mean Recall@{k}": round(np.mean(recall_list), 4),
        f"MAP@{k}": round(np.mean(map_list), 4)
    }
    df = pd.DataFrame(all_outputs)
    df.to_excel("outputs/eval_results.xlsx", index=False)
    log("Detailed results exported to outputs/eval_results.xlsx")
    return metrics
if __name__ == "__main__":
    log("Loading resources...")
    index, texts, metadata = load_resources()
    eval_set = load_eval_set("query_eval_set.json")
    log("Evaluating MAP@5 and Recall@5 with filters...")
    metrics = evaluate_map_recall(
        index, texts, metadata, eval_set,
        k=5,
        max_duration=45,
        required_types=["cognitive", "personality", "technical"]
    )
    print("\nEvaluation Metrics:")
    for key, v in metrics.items():
        print(f"{key}: {v}")
