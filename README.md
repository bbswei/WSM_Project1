##### 所有所需檔案如下：
- main.py: 主要執行檔，為英文搜索，於其中下達指令 “python main.py --Eng_query <query> --Chi_query <query>” 可檢視兩種語言的搜索成果
- Chinese.py: 為中文搜索，以jieba作為分詞套件
- PorterStemmer.py: Stemming
- Parser.py: 簡易 Preprocessing
- util.py: 提供計算的函數如cosine,euclidean distance
- Evaluation.py: 檢驗模型效能的分數回報

##### 在英文的搜索中，將會返回以五種不同評分方式所計算出的結果，如下：
- TF Weighting + Cosine Similarity
- TF-IDF Weighting + Cosine Similarity
- TF Weighting + Euclidean Distance
- TF-IDF Weighting + Euclidean Distance
- psuedo feedback
##### 而在中文的搜索中則只會包含下列兩種：
- TF Weighting + Cosine Similarity
- TF-IDF Weighting + Cosine Similarity
##### Evaluation.py 使用 TF Weighting + Cosine Similarity 分數評估，所含指標如下：
- MAA@10
- MAP@10
- Recall@10

