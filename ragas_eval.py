# import os
# import json
# from datasets import Dataset
# from sentence_transformers import SentenceTransformer
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall
# from groq import Groq

# # ----------------------------
# # CONFIG
# # ----------------------------
# os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"  # put your key here
# GROQ_MODEL = "llama3-8b-8192"
# JSON_FILE = "ragas_eval.json"

# # ----------------------------
# # LLM WRAPPER (with set_run_config)
# # ----------------------------
# class GroqLLM:
#     def __init__(self, model_name):
#         self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
#         self.model_name = model_name
#         self._run_config = None

#     def set_run_config(self, run_config):
#         """RAGAS calls this before evaluation starts."""
#         self._run_config = run_config

#     def generate(self, prompt: str) -> str:
#         resp = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
#         return resp.choices[0].message.content.strip()

# # ----------------------------
# # LOAD DATASET
# # ----------------------------
# with open(JSON_FILE, "r", encoding="utf-8") as f:
#     data = json.load(f)

# if isinstance(data, dict):
#     data = [data]

# dataset = Dataset.from_list(data)

# # ----------------------------
# # INIT MODELS
# # ----------------------------
# llm = GroqLLM(GROQ_MODEL)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ----------------------------
# # RUN EVALUATION
# # ----------------------------
# metrics = [faithfulness, answer_correctness, context_precision, context_recall]

# results = evaluate(
#     dataset,
#     metrics=metrics,
#     llm=llm,
#     embeddings=embedding_model
# )

# # ----------------------------
# # PRINT RESULTS
# # ----------------------------
# print("\n📊 RAGAS Evaluation Results:")
# print(results)




# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer, util

# # Load your data (replace with your JSON file path)
# with open(r"D:\voice_chat_rag\ragas_eval.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Load a small embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Function to compute cosine similarity
# def cosine_sim(vec1, vec2):
#     return float(util.cos_sim(vec1, vec2)[0][0])

# # Metrics storage
# faithfulness_scores = []
# answer_correctness_scores = []
# context_precision_scores = []
# context_recall_scores = []

# for item in data:
#     question = item.get("question", "")
#     answer = item.get("answer", "")
#     context = item.get("context", [])

#     # Skip if question or answer is missing
#     if not question or not answer:
#         continue

#     # Generate embeddings
#     q_emb = model.encode(question, convert_to_tensor=True)
#     a_emb = model.encode(answer, convert_to_tensor=True)

#     # Faithfulness = similarity between answer and all context combined
#     if context:
#         combined_context = " ".join(context)
#         ctx_emb = model.encode(combined_context, convert_to_tensor=True)
#         faithfulness = cosine_sim(a_emb, ctx_emb)
#     else:
#         faithfulness = 0.0

#     # Answer Correctness = similarity between answer and question
#     answer_correctness = cosine_sim(a_emb, q_emb)

#     # Context Precision = avg similarity between answer and each context snippet
#     precision_scores = []
#     if context:
#         for ctx in context:
#             ctx_emb = model.encode(ctx, convert_to_tensor=True)
#             precision_scores.append(cosine_sim(a_emb, ctx_emb))
#         context_precision = np.mean(precision_scores)
#     else:
#         context_precision = 0.0

#     # Context Recall = similarity between question and combined context
#     if context:
#         combined_context = " ".join(context)
#         ctx_emb = model.encode(combined_context, convert_to_tensor=True)
#         context_recall = cosine_sim(q_emb, ctx_emb)
#     else:
#         context_recall = 0.0

#     # Append scores
#     faithfulness_scores.append(faithfulness)
#     answer_correctness_scores.append(answer_correctness)
#     context_precision_scores.append(context_precision)
#     context_recall_scores.append(context_recall)

# # Compute overall averages
# avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
# avg_answer_correctness = np.mean(answer_correctness_scores) if answer_correctness_scores else 0.0
# avg_context_precision = np.mean(context_precision_scores) if context_precision_scores else 0.0
# avg_context_recall = np.mean(context_recall_scores) if context_recall_scores else 0.0

# # Print results
# print("\nApproximate RAG Evaluation Metrics:")
# print(f"Faithfulness: {avg_faithfulness:.3f}")
# print(f"Answer Correctness: {avg_answer_correctness:.3f}")
# print(f"Context Precision: {avg_context_precision:.3f}")
# print(f"Context Recall: {avg_context_recall:.3f}")



# eval_ragas_approx.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------- CONFIG ----------
DATA_FILE = r"D:\voice_chat_rag\ragas_eval.json"   # change to your JSON filename
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.60    # tune between 0.5-0.75 depending on strictness
# ----------------------------

def to_list_field(x):
    """Normalize field to list[str]."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(s).strip() for s in x if s and str(s).strip()]
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(x)]

def maybe_split_paragraphs(text):
    """If long ground-truth block with paragraph separators, split to smaller pieces."""
    if not text or not isinstance(text, str):
        return []
    # prefer explicit paragraph splits
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) > 1:
        return parts
    # otherwise keep as single entry
    return [text.strip()]

# load data
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"data file not found: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# load embedding model (sentence-transformers will use torch under the hood; we don't import torch explicitly)
model = SentenceTransformer(EMBED_MODEL)
embed_dim = model.get_sentence_embedding_dimension()

# containers
per_item_results = []
faithfulness_list = []
answer_correctness_list = []
context_precision_list = []
context_recall_list = []

for idx, item in enumerate(data):
    # fetch fields (robust to different key names)
    question = to_list_field(item.get("question") or item.get("query") or "") 
    question = question[0] if question else ""
    answer = to_list_field(item.get("answer") or "") 
    answer = answer[0] if answer else ""
    # contexts may be under several keys
    contexts = item.get("contexts")
    if contexts is None:
        contexts = item.get("context")
    if contexts is None:
        contexts = item.get("retrieved_chunks")
    contexts = to_list_field(contexts)

    # ground truth answer (string) and optional ground truth contexts
    gt_answer = item.get("ground_truth", "") or ""
    # If user provided ground-truth contexts separately (rare), use them; otherwise derive from ground_truth text
    gt_contexts_field = item.get("ground_truth_contexts") or item.get("gt_contexts") or None
    if gt_contexts_field:
        gt_contexts = to_list_field(gt_contexts_field)
    else:
        # try splitting ground truth block into paragraphs if it's long
        gt_contexts = maybe_split_paragraphs(gt_answer) if gt_answer else []

    # If contexts list is empty, we still allow computing answer vs ground truth correctness
    # Build embeddings (use convert_to_numpy=True)
    # if empty lists, we create zero-shaped arrays
    def embed_list(texts):
        if not texts:
            return np.zeros((0, embed_dim), dtype=np.float32)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    a_emb = embed_list([answer])            # shape (1, D) or (0,D) if empty
    q_emb = embed_list([question])
    ctx_embs = embed_list(contexts)         # shape (n_ctx, D)
    gt_ans_emb = embed_list([gt_answer])    # shape (1,D) or (0,D)
    gt_ctx_embs = embed_list(gt_contexts)   # shape (n_gt, D)

    # helper to safe cosine sim between (1,D) and (m,D) arrays
    def safe_cosine_matrix(A, B):
        """Return cosine_similarity(A, B) safely if A or B might be empty."""
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]))  # may be (0,0), (1,0), (0,3) etc.
        return cosine_similarity(A, B)

    # Answer Correctness: compare answer embedding to ground-truth answer embedding
    if a_emb.size and gt_ans_emb.size:
        answer_correctness = float(cosine_similarity(a_emb, gt_ans_emb)[0, 0])
    else:
        # fallback: use answer vs question similarity (when ground-truth answer is missing)
        answer_correctness = float(cosine_similarity(a_emb, q_emb)[0, 0]) if a_emb.size and q_emb.size else 0.0

    # Faithfulness: how much answer is supported by retrieved contexts -> mean similarity(answer, each retrieved context)
    if a_emb.size and ctx_embs.size:
        ans_ctx_sims = cosine_similarity(a_emb, ctx_embs)[0]   # shape (n_ctx,)
        faithfulness = float(np.mean(ans_ctx_sims))
    else:
        faithfulness = 0.0

    # Context Precision: fraction of retrieved contexts that match ANY ground-truth context (threshold)
    if ctx_embs.size and gt_ctx_embs.size:
        sim_matrix = safe_cosine_matrix(ctx_embs, gt_ctx_embs)  # n_ctx x n_gt
        matched_per_ctx = (sim_matrix > SIMILARITY_THRESHOLD).any(axis=1)  # bool per retrieved ctx
        context_precision = float(np.sum(matched_per_ctx) / len(contexts)) if len(contexts) > 0 else 0.0
        # Context Recall: fraction of ground-truth contexts that were retrieved (matched by any retrieved ctx)
        matched_per_gt = (sim_matrix > SIMILARITY_THRESHOLD).any(axis=0)
        context_recall = float(np.sum(matched_per_gt) / len(gt_contexts)) if len(gt_contexts) > 0 else 0.0
    else:
        context_precision = 0.0
        context_recall = 0.0

    # store
    per_item_results.append({
        "index": idx,
        "question": question,
        "answer": answer,
        "n_retrieved_contexts": len(contexts),
        "n_gt_contexts": len(gt_contexts),
        "faithfulness": round(faithfulness, 4),
        "answer_correctness": round(answer_correctness, 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4)
    })

    faithfulness_list.append(faithfulness)
    answer_correctness_list.append(answer_correctness)
    context_precision_list.append(context_precision)
    context_recall_list.append(context_recall)

# aggregate averages
def safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0

avg_faith = safe_mean(faithfulness_list)
avg_acc = safe_mean(answer_correctness_list)
avg_cprec = safe_mean(context_precision_list)
avg_crec = safe_mean(context_recall_list)

# print per-item and averages
for r in per_item_results:
    print(f"[{r['index']}] Q: {r['question']!s}")
    print(f"    faithfulness: {r['faithfulness']:.4f}, answer_correctness: {r['answer_correctness']:.4f}, "
          f"context_precision: {r['context_precision']:.4f}, context_recall: {r['context_recall']:.4f}")
print("\n--- AVERAGES ---")
print(f"Faithfulness: {avg_faith:.4f}")
print(f"Answer Correctness: {avg_acc:.4f}")
print(f"Context Precision: {avg_cprec:.4f}")
print(f"Context Recall: {avg_crec:.4f}")
