import os
import json
import time
import streamlit as st
from resume_rag import ResumeRAG, read_file
from job_matcher import JobMatcher

st.set_page_config(page_title="Resume Matcher", page_icon="📄")

st.title("📄 AI Resume & Profile Matcher")
st.write("Match candidate resumes against job descriptions using semantic search and skill matching.")

@st.cache_resource
def load_matcher():
    rag = ResumeRAG()
    return JobMatcher(rag=rag)

matcher = load_matcher()

# Sidebar options
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of matches", min_value=1, max_value=10, value=5)

if st.sidebar.button("Re-index Resumes"):
    with st.spinner("Re-indexing resumes..."):
        matcher.rag.index_resumes()
        st.sidebar.success("Done re-indexing!")

# Job Description selection
st.subheader("Job Description")

jd_folder = os.path.join("data", "job_descriptions")
sample_files = [f for f in os.listdir(jd_folder) if f.endswith(".txt")] if os.path.exists(jd_folder) else []

use_sample = st.checkbox("Use a sample job description", value=True)

if use_sample and sample_files:
    chosen_file = st.selectbox("Select Sample JD:", sample_files)
    jd_content = read_file(os.path.join(jd_folder, chosen_file))
else:
    jd_content = ""

jd_text = st.text_area("Job Description:", value=jd_content, height=180, placeholder="Paste job description here...")

# Search button
if st.button("Find Matches", type="primary"):
    if not jd_text.strip():
        st.warning("Please enter a job description first.")
    else:
        start_time = time.time()
        with st.spinner("Searching resumes..."):
            results = matcher.match_job(jd_text, top_k=top_k)
        duration = round((time.time() - start_time) * 1000, 1)

        st.success(f"Matched in {duration} ms")

        matches = results.get("top_matches", [])
        st.subheader(f"Top {len(matches)} Candidates")

        for idx, cand in enumerate(matches, 1):
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{idx}. {cand['candidate_name']}")
                with col2:
                    st.metric("Score", f"{cand['match_score']}/100")

                skills = ", ".join(cand["matched_skills"]) if cand["matched_skills"] else "None"
                st.write(f"**Matched Skills:** {skills}")
                st.write(f"**Reasoning:** {cand['reasoning']}")

                if cand.get("relevant_excerpts"):
                    with st.expander("View Resume Excerpt"):
                        for excerpt in cand["relevant_excerpts"]:
                            st.info(excerpt)

        # Output JSON as required by assignment
        with st.expander("View Output JSON Format"):
            st.json(results)
