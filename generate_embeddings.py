import os
import json
import numpy as np
import faiss
from datetime import datetime
from InstructorEmbedding import INSTRUCTOR
import torch
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
def load_assessments(json_path):
    log("Loading cleaned SHL assessment metadata...")
    with open(json_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)
    log(f"Loaded {len(assessments)} assessments.")
    return assessments
def create_textual_representation(item):
    try:
        return (
            f"{item['Assessment Name']} | "
            f"Standard Type: {item.get('Standardized Test Type', 'unknown')} | "
            f"Original Type: {item.get('Test Type', 'N/A')} | "
            f"Duration: {item.get('Duration', 'N/A')} mins | "
            f"Remote: {item.get('Remote Testing Support', 'N/A')} | "
            f"Adaptive: {item.get('Adaptive/IRT Support', 'N/A')}"
        )
    except KeyError as e:
        log(f"Warning: Missing key {e} in item: {item}")
        return ""
def prepare_texts(assessments):
    log("Preparing textual representations for embedding...")
    texts = [create_textual_representation(a) for a in assessments]
    texts = [t for t in texts if t.strip()]
    log(f"Prepared {len(texts)} valid texts.")
    return texts
def embed_texts_instructor(texts, device="cpu"):
    log(f"Loading instructor-xl on {device}...")
    model = INSTRUCTOR("hkunlp/instructor-xl")
    model.to(device)
    instruction = "Represent the task: retrieve relevant assessments based on this job description"
    input_pairs = [[instruction, text] for text in texts]
    log("Encoding all assessments with Instructor XL...")
    embeddings = model.encode(
        input_pairs,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    log(f"Generated {len(embeddings)} embeddings.")
    return embeddings
def save_outputs(embeddings, texts, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "assessment_embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "assessment_texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    log("Embeddings and texts saved.")
def save_faiss_index(embeddings, output_dir="outputs"):
    log("Fitting FAISS index...")
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.idx"))
    log("FAISS index saved.")
if __name__ == "__main__":
    # Automatically fallback to CPU if CUDA isn't available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_input_path = "shl_metadata_index_cleaned.json"  # Use cleaned metadata
    assessments = load_assessments(json_input_path)
    texts = prepare_texts(assessments)
    embeddings = embed_texts_instructor(texts, device=device)
    save_outputs(embeddings, texts)
    save_faiss_index(embeddings)
    log("[✓] All done! Embedding pipeline complete.")
