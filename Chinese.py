from pprint import pprint
from Parser import Parser
import util
import numpy as np
import pandas as pd
import os
import jieba
import argparse

class VectorSpaceChinese:

    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """
    #Collection of document term vectors
    documentVectors = {}

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]
    tfidf_vectors = {}

    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors = {}
        self.tfidf_vectors = {}
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents.values())
        self.documentVectors = {doc_name: self.makeVector(doc) for doc_name, doc in documents.items()}
        # Calculate TF-IDF vectors
        tf_values = np.array(list(self.documentVectors.values()))
        df = np.sum(tf_values > 0, axis=0)
        # to avoid value = 0, plus 1
        idf = np.log((len(documents) + 1) / (1 + df) + 1)
        tfidf_vectors = tf_values * idf

        self.tfidf_vectors={}
        for i, (doc, TFIDF) in enumerate(zip(documents.keys(), tfidf_vectors)):
            self.tfidf_vectors[doc] = TFIDF.tolist()

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        #Mapped documents into a single word string	       
        vocabularyString = " ".join(documentList)
        vocabularyList = set(jieba.cut(vocabularyString, cut_all=False))
        vectorIndex={}
        offset=0
        for word in vocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex

    def makeVector(self, wordString):
        """ Convert Chinese strings into vector """
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = set(jieba.cut(wordString, cut_all=False)) 

        for word in wordList:
            if word in self.vectorKeywordIndex:  # 確保關鍵字在索引中
                vector[self.vectorKeywordIndex[word]] += 1  # Use simple Term Count Model
        return vector
     
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query
    
    def search_tfcos(self, searchList):
        """ Search for documents that match based on a list of terms using TF and cosine """
        queryVector = self.buildQueryVector(searchList)
        ratings1 = {doc: util.cosine(queryVector, tf_vectors)
                   for doc, tf_vectors in self.documentVectors.items()}
        tf_cos_df = pd.DataFrame(list(ratings1.items()), columns=['Title', 'Score'])
        tf_cos = tf_cos_df.sort_values(by='Score', ascending=False).head(10)
        return tf_cos

    def search_tfidfcos(self, searchList):
        """ Use TF-IDF and cosine """
        queryVector = self.buildQueryVector(searchList)
        ratings2 = {doc: util.cosine(queryVector, tfidf_vectors)
                   for doc, tfidf_vectors in self.tfidf_vectors.items()}
        tfidf_cos_df = pd.DataFrame(list(ratings2.items()), columns=['Title', 'Score'])
        tfidf_cos = tfidf_cos_df.sort_values(by='Score', ascending=False).head(10)
        return tfidf_cos
    
def read_documents(folder):
    documents = {}
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):  # 確保只讀取文本文件
            with open(os.path.join(folder, filename), encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

if __name__ == '__main__':

    folder = './ChineseNews'
    documents = read_documents(folder)
    print(f"Number of documents read: {len(documents)}")
    vectorSpace = VectorSpaceChinese(documents)

    searchList = ["資安","遊戲"]

    # TF Weighting + Cosine Similarity
    ratings_tf_cos = vectorSpace.search_tfcos(searchList)
    tf_cos_df = pd.DataFrame(list(ratings_tf_cos.items()), columns=['Title', 'Score'])
    tf_cos = tf_cos_df.sort_values(by='Score', ascending=False).head(10)
    print("TF Cosine")
    print(tf_cos)

    # TF-IDF Weighting + Cosine Similarity
    ratings_tfidf_cos = vectorSpace.search_tfidfcon(searchList)
    tfidf_cos_df = pd.DataFrame(list(ratings_tfidf_cos.items()), columns=['Title', 'Score'])
    tfidf_cos = tfidf_cos_df.sort_values(by='Score', ascending=False).head(10)
    print("TF-IDF Cosine")
    print(tfidf_cos)    