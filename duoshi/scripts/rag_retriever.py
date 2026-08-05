import os
import pickle
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 清理提问套语
QUESTION_SUFFIX_PATTERN = re.compile(
    r"(?:这句(?:诗)?|该句(?:诗)?)?"
    r"(?:使用|运用|化用|引用|用)(?:了)?"
    r"(?:的)?(?:是)?(?:什么|哪个|哪一则)?典故[呢？?。！!\s]*$"
)
QUESTION_PREFIX_PATTERN = re.compile(r"^(?:请问|请分析|请判断)[，,：:\s]*")

def clean_query(query):
    query = query.strip()
    cleaned_query = QUESTION_PREFIX_PATTERN.sub('', query)
    cleaned_query = QUESTION_SUFFIX_PATTERN.sub('', cleaned_query)
    cleaned_query = cleaned_query.strip(" \t\r\n，,。！？?!；;：:")
    return cleaned_query or query

def get_allusion_confidence(candidates, chosen_idx=0):
    chosen_allusion_idx = candidates[chosen_idx]["allusion_idx"]
    chosen_scores = [cand["score"] for cand in candidates if cand["allusion_idx"] == chosen_allusion_idx]
    other_scores = [cand["score"] for cand in candidates if cand["allusion_idx"] != chosen_allusion_idx]
    chosen_score = max(chosen_scores)
    score_margin = chosen_score - max(other_scores) if other_scores else chosen_score
    return chosen_score, score_margin

# rag检索器
class RAGRetriever:
    def __init__(self,
                 index_path="./data/rag/faiss_index_allusion.bin",
                 metadata_path="./data/rag/metadata_allusion.pkl",
                 model_path="./models/fine_tuned/bge-large-zh-v1.5",
                 threshold=0.5,
                 top_k=10):
        self.threshold = threshold
        self.top_k = top_k
        self.model = SentenceTransformer(model_path)
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.index_entries = self.metadata["index_entries"]
        self.allusion_data = self.metadata["allusion_data"]
        if self.index.ntotal != len(self.index_entries):
            raise ValueError("索引与元数据数量不一致，请重新构建典故主索引。")
        if any(i != entry.get("poem_id") for i, entry in enumerate(self.index_entries)):
            raise ValueError("诗例 ID 与 FAISS 行号不一致，请重新构建典故主索引。")

    def retrieve(self, query):
        query = clean_query(query)
        q_emb = self.model.encode(query, normalize_embeddings=True)
        q_emb = np.array([q_emb]).astype('float32')

        search_k = min(len(self.index_entries), 10)
        scores_all, indices = self.index.search(q_emb, search_k)
        scores_all = scores_all[0]
        indices = indices[0]

        candidates = []
        for score, idx in zip(scores_all, indices):
            if score < self.threshold:
                continue
            entry = self.index_entries[idx]
            allu_idx = entry["allusion_idx"]
            candidates.append({
                "score": float(score),
                "poem_id": entry["poem_id"],
                "matched_poem": entry["poem_text"],
                "allusion_idx": allu_idx,
                "allusion": self.allusion_data[allu_idx]
            })

        return candidates[:self.top_k]

# 单例模式
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
