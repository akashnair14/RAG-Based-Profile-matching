import os
import re
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "resumes")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Read text or PDF files
def read_file(file_path):
    if file_path.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

# Extract basic details from resume text
def extract_metadata(text):
    # Name
    name_match = re.search(r"NAME:\s*(.+)", text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else "Unknown"

    # Years of experience
    exp_match = re.search(r"(?:EXPERIENCE\s*YEARS|YEARS\s*OF\s*EXPERIENCE|EXPERIENCE):\s*(\d+)", text, re.IGNORECASE)
    if exp_match:
        years_exp = int(exp_match.group(1))
    else:
        all_years = re.findall(r"(\d+)\+?\s*years?", text, re.IGNORECASE)
        years_exp = max([int(y) for y in all_years if int(y) < 40], default=0)

    # Education
    edu_match = re.search(r"EDUCATION:\s*\n?(?:-\s*)?([^\n]+)", text, re.IGNORECASE)
    education = edu_match.group(1).strip() if edu_match else "Not specified"

    # Skills
    skills_match = re.search(r"SKILLS:\s*\n?([^\n]+)", text, re.IGNORECASE)
    if skills_match:
        skills = [s.strip() for s in skills_match.group(1).split(",") if s.strip()]
    else:
        skills = []

    return {
        "candidate_name": name,
        "experience_years": years_exp,
        "education": education,
        "skills": skills
    }

# Split resume into section chunks (Experience, Education, Skills, etc.)
def chunk_resume(text):
    sections = re.split(r"(?=(?:NAME|SUMMARY|SKILLS|EXPERIENCE|EDUCATION|PROJECTS):\s*)", text, flags=re.IGNORECASE)
    chunks = []
    for sec in sections:
        cleaned = sec.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks if chunks else [text.strip()]

# RAG class for managing vector database
class ResumeRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        # Check if OpenRouter / OpenAI API key is available
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if api_key and not api_key.startswith("your_"):
            try:
                self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-small",
                    api_base=base_url if base_url else None
                )
            except Exception:
                self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        else:
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        self.collection = self.client.get_or_create_collection(
            name="resumes_collection",
            embedding_function=self.embed_fn
        )

    def index_resumes(self):
        # Clear existing data first
        try:
            self.client.delete_collection("resumes_collection")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="resumes_collection",
            embedding_function=self.embed_fn
        )

        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf"))]
        print(f"Found {len(files)} resumes to index...")

        doc_id = 0
        for file_path in files:
            text = read_file(file_path)
            meta = extract_metadata(text)
            chunks = chunk_resume(text)

            docs = []
            metadatas = []
            ids = []

            for chunk in chunks:
                doc_id += 1
                docs.append(chunk)
                metadatas.append({
                    "candidate_name": meta["candidate_name"],
                    "resume_path": file_path,
                    "experience_years": meta["experience_years"],
                    "education": meta["education"],
                    "skills_str": ", ".join(meta["skills"])
                })
                ids.append(f"doc_{doc_id}")

            if docs:
                self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
            print(f"Indexed: {meta['candidate_name']} ({meta['experience_years']} yrs)")

        print("All resumes indexed successfully!")

if __name__ == "__main__":
    rag = ResumeRAG()
    rag.index_resumes()
