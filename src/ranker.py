import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from src.normalizer import ProfileNormalizer

class SemanticTalentMatcher:
    """Employs a two-tier retrieval and ranking cascade to match talent conceptually."""
    
    def __init__(self, retrieval_model: str = 'BAAI/bge-large-en-v1.5', reranker_model: str = 'ms-marco-MiniLM-L-6-v2'):
        print(f"[Engine] Activating Tier-1 Retrieval Vector Spaces ({retrieval_model})...")
        self.bi_encoder = SentenceTransformer(retrieval_model)
        
        print(f"[Engine] Activating Tier-2 Contextual Validation Layers ({reranker_model})...")
        self.cross_encoder = CrossEncoder(reranker_model)

    def process(self, jd_data: dict, candidates_data: list, top_n: int = 50) -> pd.DataFrame:
        # 1. Standardize text strings using the normalizer
        jd_text = ProfileNormalizer.flatten_jd(jd_data)
        profile_texts = [ProfileNormalizer.flatten_candidate(c) for c in candidates_data]
        
        print(f"[Processing] Transforming {len(profile_texts)} candidates into multi-dimensional vectors...")
        jd_vector = self.bi_encoder.encode(jd_text, convert_to_numpy=True).reshape(1, -1)
        profile_vectors = self.bi_encoder.encode(profile_texts, convert_to_numpy=True)
        
        # Calculate raw matrix cosine distances
        retrieval_scores = cosine_similarity(jd_vector, profile_vectors).flatten()
        
        dataset = pd.DataFrame({
            "candidate_id": [c.get("candidate_id") for c in candidates_data],
            "name": [c.get("name", "Hidden Identity") for c in candidates_data],
            "serialized_profile": profile_texts,
            "retrieval_score": retrieval_scores
        })
        
        eval_window = min(len(dataset), top_n * 3)
        shortlist_pool = dataset.nlargest(eval_window, "retrieval_score").copy()
        
        print(f"[Processing] Running deep contextual evaluations on top {eval_window} candidates...")
        cross_inputs = [[jd_text, profile] for profile in shortlist_pool["serialized_profile"].tolist()]
        
        rerank_scores = self.cross_encoder.predict(cross_inputs)
        shortlist_pool["raw_affinity_score"] = rerank_scores
        
        final_rankings = shortlist_pool.sort_values(by="raw_affinity_score", ascending=False).head(top_n).copy()
        
        calibration_factor = 0.5  
        final_rankings["match_score_pct"] = 100 / (1 + np.exp(-calibration_factor * final_rankings["raw_affinity_score"]))
        
        final_rankings["match_score_pct"] = final_rankings["match_score_pct"].round(1).astype(int)
        final_rankings["rank"] = range(1, len(final_rankings) + 1)
        
        return final_rankings[["rank", "candidate_id", "name", "match_score_pct"]]