import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download

# 配置
EXCEL_PATH = "./data/raw/chuxueji_with_poem_example.xlsx"
BGE_MODEL_PATH = "./models/fine_tuned/bge-large-zh-v1.5"
INDEX_DIR = "./data/rag"
POEM_WEIGHT = 0.75
NAME_WEIGHT = 0.05
VARIANT_WEIGHT = 0.05
TAG_WEIGHT = 0.05
SOURCE_WEIGHT = 0.10
os.makedirs(INDEX_DIR, exist_ok=True)

df = pd.read_excel(EXCEL_PATH)

text_columns = ['allusion_name', 'source_text', 'allusion_mean', 'semantic_tags', 'allusion_variants', 'poem_example']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str)

print(f"读取到 {len(df)} 条典故记录。")
allusion_data = df.to_dict(orient='records')

# 构建诗例级主索引文本
index_entries = []
for idx, row in enumerate(allusion_data):
    poem_example = row.get('poem_example', '').strip()
    if not poem_example:
        continue

    poem_lines = [line.strip() for line in poem_example.split('\n') if line.strip()]
    for poem in poem_lines:
        poem_id = len(index_entries)
        search_fields = [
            ("诗例", poem),
            ("典故名", row.get('allusion_name', '')),
            ("异名", row.get('allusion_variants', '')),
            ("语义标签", row.get('semantic_tags', '')),
            ("原文", row.get('source_text', '')),
        ]
        search_text = "\n".join(f"{name}：{value}" for name, value in search_fields if value.strip())
        index_entries.append({
            "poem_id": poem_id,
            "poem_text": poem,
            "search_text": search_text,
            "allusion_idx": idx
        })

print(f"共构建 {len(index_entries)} 条诗例级主索引文本。")

# 加载 BGE 模型
if not os.path.exists(BGE_MODEL_PATH):
    print("下载 BGE 模型...")
    snapshot_download('BAAI/bge-large-zh-v1.5', local_dir=BGE_MODEL_PATH)
model = SentenceTransformer(BGE_MODEL_PATH)

print("编码诗例向量...")
poem_texts = [entry["poem_text"] for entry in index_entries]
poem_embeddings = model.encode(poem_texts, normalize_embeddings=True, show_progress_bar=True)
poem_embeddings = np.array(poem_embeddings).astype('float32')
dim = poem_embeddings.shape[1]

def encode_optional_texts(texts, field_name):
    embeddings = np.zeros((len(texts), dim), dtype='float32')
    valid_indices = [i for i, text in enumerate(texts) if text.strip()]
    if not valid_indices:
        return embeddings

    print(f"编码{field_name}向量...")
    valid_texts = [texts[i] for i in valid_indices]
    valid_embeddings = model.encode(valid_texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings[valid_indices] = np.array(valid_embeddings).astype('float32')
    return embeddings

allusion_indices = [entry["allusion_idx"] for entry in index_entries]
name_texts = [allusion_data[i].get('allusion_name', '') for i in allusion_indices]
variant_texts = [allusion_data[i].get('allusion_variants', '') for i in allusion_indices]
tag_texts = [allusion_data[i].get('semantic_tags', '') for i in allusion_indices]
source_texts = [allusion_data[i].get('source_text', '') for i in allusion_indices]

name_embeddings = encode_optional_texts(name_texts, "典故名")
variant_embeddings = encode_optional_texts(variant_texts, "异名")
tag_embeddings = encode_optional_texts(tag_texts, "语义标签")
source_embeddings = encode_optional_texts(source_texts, "原文")

allusion_embeddings = (
    POEM_WEIGHT * poem_embeddings
    + NAME_WEIGHT * name_embeddings
    + VARIANT_WEIGHT * variant_embeddings
    + TAG_WEIGHT * tag_embeddings
    + SOURCE_WEIGHT * source_embeddings
)
norms = np.linalg.norm(allusion_embeddings, axis=1, keepdims=True)
allusion_embeddings = allusion_embeddings / np.maximum(norms, 1e-12)
allusion_embeddings = allusion_embeddings.astype('float32')

# 构建 Faiss 索引
index = faiss.IndexFlatIP(dim)
index.add(allusion_embeddings)
print(f"诗例级主索引维度 {dim}，共 {len(index_entries)} 条。")

faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index_allusion.bin"))
with open(os.path.join(INDEX_DIR, "metadata_allusion.pkl"), "wb") as f:
    pickle.dump({
        "index_entries": index_entries,
        "allusion_data": allusion_data,
        "dim": dim,
        "field_weights": {
            "poem": POEM_WEIGHT,
            "name": NAME_WEIGHT,
            "variant": VARIANT_WEIGHT,
            "tag": TAG_WEIGHT,
            "source": SOURCE_WEIGHT
        }
    }, f)
print(f"典故主索引和元数据已保存到 {INDEX_DIR}")
