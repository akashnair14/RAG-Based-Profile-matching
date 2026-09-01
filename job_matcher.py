import os
import re
import json
from resume_rag import ResumeRAG, read_file

class JobMatcher:
    def __init__(self, rag=None):
        self.rag = rag if rag else ResumeRAG()
        self.collection = self.rag.collection

    # Extract required skills and experience from JD
    def parse_job_description(self, jd_text):
        exp_match = re.search(r"(\d+)\+?\s*years?", jd_text, re.IGNORECASE)
        min_years = int(exp_match.group(1)) if exp_match else 0

        # Extract skills mentioned in requirements
        common_skills = [
            "Python", "PyTorch", "TensorFlow", "LangChain", "ChromaDB", "Pinecone", "RAG", "LLMs",
            "React", "TypeScript", "JavaScript", "Next.js", "Redux", "Tailwind CSS", "Node.js",
            "FastAPI", "Django", "PostgreSQL", "Redis", "Kafka", "Docker", "Kubernetes", "AWS",
            "Terraform", "CI/CD", "Spark", "Snowflake", "BigQuery", "Tableau", "PowerBI", "Pandas"
        ]
        
        required_skills = []
        for skill in common_skills:
            if re.search(rf"\b{re.escape(skill)}\b", jd_text, re.IGNORECASE):
                required_skills.append(skill)

        return min_years, required_skills

    # Match resumes against JD
    def match_job(self, jd_input, top_k=10):
        # Read file if a path is given
        if os.path.isfile(jd_input):
            jd_text = read_file(jd_input)
        else:
            jd_text = str(jd_input)

        min_years, required_skills = self.parse_job_description(jd_text)

        # 1. Semantic Search using ChromaDB
        results = self.collection.query(
            query_texts=[jd_text],
            n_results=top_k * 3
        )

        candidates = {}
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.5] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                name = meta["candidate_name"]
                # Convert distance to similarity score
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                if name not in candidates:
                    skills_list = [s.strip() for s in meta["skills_str"].split(",") if s.strip()]
                    candidates[name] = {
                        "candidate_name": name,
                        "resume_path": meta["resume_path"],
                        "experience_years": meta["experience_years"],
                        "skills": skills_list,
                        "best_similarity": similarity,
                        "excerpts": []
                    }
                else:
                    if similarity > candidates[name]["best_similarity"]:
                        candidates[name]["best_similarity"] = similarity

                if len(candidates[name]["excerpts"]) < 2:
                    candidates[name]["excerpts"].append(doc.strip())

        # 2. Hybrid Scoring (Semantic + Skill match)
        scored_list = []
        for name, data in candidates.items():
            # Semantic score (0-50)
            semantic_score = data["best_similarity"] * 50

            # Keyword / Skill match score (0-50)
            matched = []
            if required_skills:
                for req in required_skills:
                    for cand_skill in data["skills"]:
                        if req.lower() in cand_skill.lower() or cand_skill.lower() in req.lower():
                            if req not in matched:
                                matched.append(req)
                skill_score = (len(matched) / len(required_skills)) * 50
            else:
                matched = data["skills"][:4]
                skill_score = 35

            # Calculate total score out of 100
            total_score = int(round(semantic_score + skill_score))

            # Apply experience filter penalty if candidate has less experience than required
            if min_years > 0 and data["experience_years"] < min_years:
                total_score = int(total_score * 0.8)

            # Cap score between 0 and 100
            total_score = max(0, min(100, total_score))

            # Generate simple reasoning
            skills_text = ", ".join(matched[:4]) if matched else "general technical skills"
            if total_score >= 75:
                reasoning = f"Strong match with {data['experience_years']} years experience and matching skills: {skills_text}."
            elif total_score >= 60:
                reasoning = f"Good match with {data['experience_years']} years experience. Overlaps in {skills_text}."
            else:
                reasoning = f"Moderate fit with {data['experience_years']} years experience. Partial skill match in {skills_text}."

            scored_list.append({
                "candidate_name": data["candidate_name"],
                "resume_path": data["resume_path"],
                "match_score": total_score,
                "matched_skills": matched,
                "relevant_excerpts": data["excerpts"][:2],
                "reasoning": reasoning
            })

        # Sort by match score descending
        scored_list.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "job_description": jd_text[:200] + ("..." if len(jd_text) > 200 else ""),
            "top_matches": scored_list[:top_k]
        }

if __name__ == "__main__":
    matcher = JobMatcher()
    sample_jd = os.path.join("data", "job_descriptions", "jd_1_senior_ml_engineer.txt")
    print("--- Testing Matcher ---")
    result = matcher.match_job(sample_jd, top_k=5)
    print(json.dumps(result, indent=2))
