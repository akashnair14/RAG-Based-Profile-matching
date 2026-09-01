# RAG-Based Resume & Profile Matching Engine

An intelligent, lightweight Retrieval-Augmented Generation (RAG) profile matching engine designed to match candidate resumes against job descriptions using semantic search, hybrid keyword scoring, and metadata filtering.

## Key Features
- Intelligent Document Chunking: Splits resumes preserving logical sections (Education, Work Experience, Skills, Summary, Projects).
- Vector Embeddings & ChromaDB: Embeds candidate sections using local sentence transformers or OpenAI embeddings, stored in a persistent local ChromaDB instance.
- Hybrid Matching Engine: Combines dense semantic similarity with critical skill keyword overlap and experience constraint penalties (0-100 scale).
- Structured JSON Output: Returns candidate matches with matched skills, relevant excerpts, match score, and reasoning.
- Zero Cloud Setup Required: Works 100% locally out-of-the-box, or connect your OpenAI / OpenRouter API keys via .env.

## Project Structure
- data/resumes/: 32 diverse candidate resumes across multiple domains
- data/job_descriptions/: 5 comprehensive job descriptions (ML, Frontend, Backend, Data, DevOps)
- src/config.py: Configuration and path management
- src/utils.py: Document loading, metadata extraction, and section chunker
- src/resume_rag.py: Part A - Vector store and indexing pipeline
- src/job_matcher.py: Part B - Hybrid retrieval, scoring, and reasoning engine
- chroma_db/: Persistent local ChromaDB vector store
- evaluation.ipynb: Notebook with experimentation, latency benchmarks, and visual plots
- requirements.txt: Project dependencies

## Installation & Setup
1. pip install -r requirements.txt
2. (Optional) Set OPENAI_API_KEY in .env

## Usage Guide
- Ingest and Index Resumes: python -m src.resume_rag
- Match a Job Description: python -m src.job_matcher
- Run Experiments: jupyter notebook evaluation.ipynb
