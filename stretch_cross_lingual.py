import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 1. إعداد النموذج (mBERT)
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text_list):
    # Tokenization
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=128)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean Pooling (المطلوب في الواجب)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# 2. تحميل البيانات المختارة (استخدم الـ IDs المذكورة في الجدول أعلاه)
en_ids = [1, 2, 4, 48, 27, 72, 36, 24, 42, 62]
ar_ids = [79, 80, 82, 126, 105, 150, 114, 102, 120, 140]

df = pd.read_csv("data/climate_articles.csv")
en_texts = df[df['id'].isin(en_ids)]['text'].tolist()
ar_texts = df[df['id'].isin(ar_ids)]['text'].tolist()

# 3. استخراج الـ Embeddings وحساب التشابه
en_embeddings = get_embeddings(en_texts)
ar_embeddings = get_embeddings(ar_texts)

# مصفوفة التشابه 20x20 (عن طريق دمج النصوص)
all_embeddings = torch.cat([en_embeddings, ar_embeddings], dim=0)
sim_matrix = cosine_similarity(all_embeddings)

# 4. رسم الـ Heatmap
labels = [t[:40] for t in en_texts] + [t[:40] for t in ar_texts]
plt.figure(figsize=(12, 10))
sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, cmap="YlGnBu", annot=False)
plt.title("Cross-Lingual Embedding Similarity (EN vs AR)")
plt.savefig("heatmap.png")
plt.show()