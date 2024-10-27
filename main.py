from pprint import pprint
from Parser import Parser
from Chinese import VectorSpaceChinese
import nltk
from nltk.tokenize import word_tokenize
import util
import numpy as np
import pandas as pd
import os
import argparse

class VectorSpace:

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
        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        
        vocabularyString = " ".join(documentList)
        # vocabularyString+=query
        # print(vocabularyString)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        # print(vectorIndex)
        return vectorIndex  #(keyword:position)
    
    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            if word in self.vectorKeywordIndex:  # 確保關鍵字在索引中
                vector[self.vectorKeywordIndex[word]] += 1  # Use simple Term Count Model
        return vector
    
     
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query

    #def related(self,documentId):
        #""" find documents that are related to the document indexed by passed Id within the document Vectors"""
        #ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        ##ratings.sort(reverse=True)
        #return ratings
    
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
        ratings2 = {doc: util.cosine(queryVector, tfidf_vectors)  # Remove file extension
                    for doc, tfidf_vectors in self.tfidf_vectors.items()}
        return ratings2


    def search_tfeuc(self, searchList):
        """ Use TF and Euclidean Distance """
        queryVector = self.buildQueryVector(searchList)
        ratings3 = {doc: util.euclideandistance(queryVector, tf_vectors)
                   for doc, tf_vectors in self.documentVectors.items()}
        
        tf_euc_df = pd.DataFrame(list(ratings3.items()), columns=['Title', 'Score'])
        tf_euc = tf_euc_df.sort_values(by='Score', ascending=False).head(10)
        return tf_euc
    
    def search_tfidfeuc(self, searchList):
        """ Use TF-IDF and Euclidean Distance """
        queryVector = self.buildQueryVector(searchList)
        ratings4 = {doc: util.euclideandistance(queryVector, tfidf_vectors) for doc, tfidf_vectors in self.tfidf_vectors.items()}
            
        tfidf_euc_df = pd.DataFrame(list(ratings4.items()), columns=['Title', 'Score'])
        tfidf_euc = tfidf_euc_df.sort_values(by='Score', ascending=False).head(10)
        return tfidf_euc
    
    def extract_nouns_and_verbs(self, text):
        """Extract nouns and verbs from the given text."""
        tokens = word_tokenize(text)
        tagged_text = nltk.pos_tag(tokens)
        nouns = [word for word, tag in tagged_text if tag.startswith('NN')]
        verbs = [word for word, tag in tagged_text if tag.startswith('VB')]
        return nouns, verbs
    
    def pseudo_feedback(self, query, ratings):
        most_relevant_doc = ratings.iloc[0]['Title']
        #most_relevant_doc = list(ratings.items())[0][0]
        print(f"Most relevant document: {most_relevant_doc}")  # Debug statement
        feedback_doc = documentsE.get(most_relevant_doc)
        print(f"Feedback document: {feedback_doc is not None}")  # Check if feedback_doc is found

        if feedback_doc is None:
            raise KeyError(f"Document {most_relevant_doc} not found in documentsE.")
        
        feedback_nouns, feedback_verbs = self.extract_nouns_and_verbs(feedback_doc)
        new_query_vector = self.buildPseudoQueryVector(query, feedback_nouns + feedback_verbs)
        
        ratings_pseudo = {doc: util.euclideandistance(new_query_vector, tfidf_vectors) for doc, tfidf_vectors in self.tfidf_vectors.items()}
        pseudo_df = pd.DataFrame(list(ratings_pseudo.items()), columns=['Title', 'Score'])
        p_tfidf_euc = pseudo_df.sort_values(by='Score', ascending=False).head(10)

        return p_tfidf_euc
    
    def buildPseudoQueryVector(self, termList, feedback_terms):
        feedback_vector = self.makeVector(" ".join(feedback_terms))
        
        original_query_vector = self.buildQueryVector(termList)
        pseudoQueryVector = [x + 0.5 * y for x, y in zip(original_query_vector, feedback_vector)]
        
        return pseudoQueryVector


def read_Eng_documents(folder1):
    documentsE = {}
    for filename in os.listdir(folder1):
        if filename.endswith('.txt'):  # 確保只讀取文本文件
            with open(os.path.join(folder1, filename), encoding='utf-8') as file:
                documentsE[filename] = file.read()
    return documentsE

def read_Chi_documents(folder2):
    documentsC = {}
    for filename in os.listdir(folder2):
        if filename.endswith('.txt'): 
            with open(os.path.join(folder2, filename), encoding='utf-8') as file:
                documentsC[filename] = file.read()
    return documentsC

 
if __name__ == '__main__':

    folder1 = './EnglishNews'
    folder2 = './ChineseNews'
    documentsE = read_Eng_documents(folder1)
    print(list(documentsE.keys())[:10])
    documentsC = read_Chi_documents(folder2)
    print(f"Number of English documents read: {len(documentsE)}")
    print(f"Number of Chinese documents read: {len(documentsC)}")

    english_vector_space = VectorSpace(documentsE) 
    chinese_vector_space = VectorSpaceChinese(documentsC)

    parser = argparse.ArgumentParser(description='Rankings in two languages')
    parser.add_argument('--Eng_query', nargs='+', help='English query')
    parser.add_argument('--Chi_query', nargs='+', help='Chinese query')
    args = parser.parse_args()
    print(args.Eng_query)

    if args.Eng_query:
        english_results = english_vector_space.search_tfcos(args.Eng_query)  # Use TF Cosine for English

        english_tfidf = english_vector_space.search_tfidfcos(args.Eng_query)  # Use TF-IDF Cosine for English        
        tfidf_cos_df = pd.DataFrame(list(english_tfidf.items()), columns=['Title', 'Score'])
        tfidf_cos = tfidf_cos_df.sort_values(by='Score', ascending=False).head(10)

        english_results_tfeuc = english_vector_space.search_tfeuc(args.Eng_query)  # Use TF Euclidean for English

        english_results_tfidfeuc = english_vector_space.search_tfidfeuc(args.Eng_query)  # Use TF-IDF Euclidean for English

        pseudo_results = english_vector_space.pseudo_feedback(args.Eng_query, english_results_tfidfeuc)

        print("English Search Results:")
        print("TF Cosine:")
        print(english_results)
        print("TF-IDF Cosine:")
        print(tfidf_cos)
        print("TF Euclidean:")
        print(english_results_tfeuc)
        print("TF-IDF Euclidean:")
        print(english_results_tfidfeuc)
        print("Pseudo Feedback:")
        print(pseudo_results)


    if args.Chi_query:
        chinese_results = chinese_vector_space.search_tfcos(args.Chi_query) 
        chinese_tfidf = chinese_vector_space.search_tfidfcos(args.Chi_query)    
        ctfidf_cos_df = pd.DataFrame(list(chinese_tfidf.items()), columns=['Title', 'Score'])
        ctfidf_cos = tfidf_cos_df.sort_values(by='Score', ascending=False).head(10)

        print("Chinese Search Results:")
        print("TF Cosine:")
        print(chinese_results)
        print("TF-IDF Cosine:")
        print(ctfidf_cos)