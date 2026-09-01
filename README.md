# RAG-Based Resume & Profile Matching Engine

A simple Python project that matches candidate resumes against job descriptions using semantic vector search (ChromaDB) and keyword skill matching.

---

## What Does This Project Do?

Instead of manually reading dozens of resumes or relying only on exact keyword searches:
1. **Reads Resumes:** Loads text/PDF resumes and splits them into sections (Summary, Skills, Experience, Education).
2. **Converts to Embeddings:** Stores resume text in a local vector database (**ChromaDB**).
3. **Smart Matching:** Takes a job description and finds the top matching candidates based on semantic meaning, required skills, and years of experience.
4. **Scoring & Reasoning:** Gives each candidate a match score (0–100) and explains why they matched.

---

## Project Structure

```text
├── data/
│   ├── resumes/            # Candidate resume files (.txt)
│   └── job_descriptions/   # Sample job descriptions (.txt)
├── chroma_db/              # Local vector database (auto-created)
├── resume_rag.py           # Part A: Document loading, chunking & vector indexing
├── job_matcher.py          # Part B: Search, skill matching & scoring engine
├── app.py                  # Streamlit web UI
├── evaluation.ipynb        # Jupyter notebook for testing & metrics
├── requirements.txt        # Project dependencies
├── .env.example            # Sample environment file
└── README.md
```

---

## Step-by-Step Setup Guide

### 1. Clone the repository
```bash
git clone https://github.com/akashnair14/RAG-Based-Profile-matching.git
cd RAG-Based-Profile-matching
```

### 2. Install dependencies
Make sure you have Python 3.9+ installed. Run:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set up API Keys
The project works completely free and locally using `sentence-transformers`.

If you want to use OpenAI or OpenRouter embeddings instead:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your API key inside `.env`:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

---

## How to Run

### Option 1: Web Interface (Streamlit)
The easiest way to use the matcher is through the browser UI:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser. You can select a sample job description or paste your own to see matching candidates immediately.

---

### Option 2: Command Line

1. **Index all resumes into ChromaDB:**
   ```bash
   python resume_rag.py
   ```

2. **Run the job matcher on a sample job description:**
   ```bash
   python job_matcher.py
   ```

---

### Option 3: Jupyter Notebook
To see experiments, search latency, and score breakdowns:
```bash
jupyter notebook evaluation.ipynb
```

---

## Sample JSON Output

When matching a job description, the system outputs structured results like this:

```json
{
  "job_description": "JOB TITLE: Senior Machine Learning Engineer (RAG & LLMs)...",
  "top_matches": [
    {
      "candidate_name": "Dr. Sarah Chen",
      "resume_path": "data/resumes/sarah_chen.txt",
      "match_score": 81,
      "matched_skills": [
        "Python",
        "PyTorch",
        "LangChain",
        "ChromaDB",
        "RAG",
        "LLMs",
        "Docker"
      ],
      "relevant_excerpts": [
        "Staff ML Engineer with 8+ years experience designing scalable deep learning systems..."
      ],
      "reasoning": "Strong match with 8 years experience and matching skills: Python, PyTorch, LangChain, ChromaDB."
    }
  ]
}
```