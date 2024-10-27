from main import VectorSpace
import pandas as pd
import numpy as np
import os
from itertools import chain

def read_documents(folder1):
    documents = {}
    for filename in sorted(os.listdir(folder1)):
        if filename.endswith('.txt'):  # 確保只讀取文本文件
            with open(os.path.join(folder1, filename), encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

def MRR(ranking, relevant_documents):
    relevant_set = set(relevant_documents)  # 將相關文件轉換為集合
    for rank, doc_id in enumerate(ranking):
        if doc_id in relevant_set:
            return 1 / (rank + 1)
    return 0

def MAP(ranking, relevant_documents):
    precisions = []
    for rank, doc_id in enumerate(ranking):
        if doc_id in relevant_documents:
            precision = len(set(relevant_documents[:rank + 1]) & set([doc_id])) / (rank + 1)
            precisions.append(precision)

    if len(precisions) == 0:
        return 0
    return sum(precisions) / len(relevant_documents)

def Recall(ranking, relevant_documents, k=10):
    if len(relevant_documents) == 0:
        return 0 
    retrieved_relevant = set(ranking[:k]) & set(relevant_documents)
    return len(retrieved_relevant) / len(relevant_documents)

def evaluate_metrics(ranking, relevant_documents):
    mrr = MRR(ranking, relevant_documents)
    map_score = MAP(ranking, relevant_documents)
    recall = Recall(ranking, relevant_documents)

    return mrr, map_score, recall


docs_data = read_documents('./smaller_dataset/collections')
query_data = read_documents('./smaller_dataset/queries')
print(f"Number of documents read: {len(docs_data)}")
print(f"Number of queries read: {len(query_data)}")

relevant_df = pd.read_csv('./smaller_dataset/rel.tsv', sep='\t', header=None, names=['Query', 'Documents'])
relevant_documents = relevant_df.groupby('Query')['Documents'].apply(list).to_dict()

for key in relevant_documents:
    # 使用 eval 解析字符串，將字符串列表轉換為數字列表
    relevant_documents[key] = [int(doc_id) for doc_id in eval(relevant_documents[key][0])]

docs = VectorSpace(docs_data)
docs.build(docs_data)
#similarity_matrix = np.array(list(similarity.values())) #numpy陣列

# Evaluate metrics
mrr_scores = []
map_scores = []
recall_scores = []

for query_id, query_text in query_data.items():
    # Remove '.txt' from query_id for matching
    base_query_id = query_id.replace('.txt', '')

    similarity = docs.search_tfidfcos(query_text)  
    similarity_matrix = np.array(list(similarity.values()))

    ranking = [int(item[0]) for item in sorted(enumerate(similarity_matrix), key=lambda x: x[1], reverse=True)][:10]
    #ranking = [item[0] for item in sorted(enumerate(similarity_matrix), key=lambda x: x[1], reverse=True)]
    #ranking = ranking[:10]
    
    # Use base_query_id to find relevant documents
    relevant_docs = relevant_documents.get(base_query_id, [])
    relevant_docs = relevant_documents.get(base_query_id, [])

    mrr, map_score, recall = evaluate_metrics(ranking, relevant_docs)
    mrr_scores.append(mrr)
    map_scores.append(map_score)
    recall_scores.append(recall)

mean_mrr = np.mean(mrr_scores)
mean_map = np.mean(map_scores)
mean_recall = np.mean(recall_scores)

print("MRR@10:", mean_mrr)
print("MAP@10:", mean_map)
print("Recall@10:", mean_recall)