'''import streamlit as st
from handle_query import embed_query_instructor, load_index_and_metadata

st.title("SHL Assessment Recommender")

query = st.text_input("Enter your job role or need:")
if st.button("Recommend"):
    index_path = "outputs/faiss_index.idx"
    texts_path = "outputs/assessment_texts.json"
    metadata_path = "shl_metadata_index_cleaned.json"

    index, texts, metadata = load_index_and_metadata(index_path, texts_path, metadata_path)
    embedding = embed_query_instructor(query)
    
    # Faiss search
    import faiss
    D, I = index.search(embedding, k=5)
    
    st.write("Top Recommendations:")
    for i in I[0]:
        st.markdown(f"- **{metadata[i]['title']}**")'''
import streamlit as st
import torch
from handle_query import embed_query_instructor, load_index_and_metadata, search_similar_fuzzy
import trafilatura

# ========== Page Setup ==========
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommendation System")

# ========== Input Form ==========
with st.form(key="query_form"):
    job_input = st.text_area("Paste Job Description or URL:", height=200)
    max_duration = st.slider("Max Duration (minutes)", min_value=10, max_value=120, value=45, step=5)
    assessment_types = st.multiselect(
        "Preferred Assessment Types",
        ["technical", "cognitive", "personality", "language", "behavioral"],
        default=["technical", "cognitive", "personality"]
    )
    submitted = st.form_submit_button("Get Recommendations")

# ========== Logic ==========
if submitted:
    if not job_input.strip():
        st.warning("Please enter a job description or URL.")
        st.stop()

    with st.spinner("🔎 Processing input..."):
        if job_input.startswith("http"):
            text = trafilatura.extract(trafilatura.fetch_url(job_input))
            if not text:
                st.error("❌ Failed to extract content from the URL.")
                st.stop()
        else:
            text = job_input

        device = "cuda" if torch.cuda.is_available() else "cpu"
        query_vec = embed_query_instructor(text, device=device)

        try:
            index, texts, metadata = load_index_and_metadata(
                "outputs/faiss_index.idx",
                "outputs/assessment_texts.json",
                "shl_metadata_index_cleaned.json"
            )
        except FileNotFoundError:
            st.error("❌ Required index or metadata files not found in 'outputs/' folder.")
            st.stop()

        results = search_similar_fuzzy(
            query_vec, index, metadata,
            top_k=10,
            max_duration=max_duration,
            required_types=assessment_types
        )

    # ========== Output ==========
    st.subheader("📋 Top Recommended Assessments")
    if results:
        for name, url, score in results:
            st.markdown(f"**[{name}]({url})**  \nSimilarity Score: `{score:.4f}`")
    else:
        st.info("No matching assessments found with the current filters.")

