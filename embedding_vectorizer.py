from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, BaseEstimator
import numpy as np
import abc
from scipy.sparse import csr_matrix, hstack, csc_matrix

class EmbeddingVectorizer(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, word2vec, bow_vectorizer=None):
        self.word2vec = word2vec
        self.dim = word2vec.vector_size
        self.bow_vectorizer = bow_vectorizer

    def fit(self, X, y=None):
        if self.bow_vectorizer is not None:
            self.bow_vectorizer.fit_transform(X, y)
        return self

    def transform(self, X):
        docs_matrix = []
        for document in X:
            document = document.lower()
            document = document.replace('ne_', 'NE_')
            word_vectors = []
            for word in document.split():
                if self.word_known(word):
                    word_vectors.append(np.array(self.word2vec[word] * self.word_weight(word)))
            if word_vectors:
                doc_vector = np.mean(np.array(word_vectors), axis=0)
            else:
                doc_vector = np.zeros(self.dim)
            docs_matrix.append(doc_vector)
        X_word2vec = np.array(docs_matrix)
        if self.bow_vectorizer is None:
            return docs_matrix
        else:
            X_transformed = self.bow_vectorizer.transform(X)
            X_transformed = X_transformed.astype(np.float32)
            X_word2vec = csr_matrix(X_word2vec)
            return hstack((X_transformed, X_word2vec), format='csr')


    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)

    @abc.abstractmethod
    def word_known(self, word):
        """Method that checks whether the given word exists in the model"""
        return

    @abc.abstractmethod
    def word_weight(self, word):
        """Method that returns a weight for the vector of the given word"""
        return


class MeanEmbeddingVectorizer(EmbeddingVectorizer):
    def word_known(self, word):
        return word in self.word2vec

    def word_weight(self, word):
        return 1

class MeanEmbeddingVectorizerInNegatedText(MeanEmbeddingVectorizer):
    def transform(self, X):
        docs_matrix = []
        for document in X:
            document = document.lower()
            document = document.replace('ne_', '')
            word_vectors = []
            for word in document.split():
                if self.word_known(word):
                    word_vectors.append(np.array(self.word2vec[word] * self.word_weight(word)))
            if word_vectors:
                doc_vector = np.mean(np.array(word_vectors), axis=0)
            else:
                doc_vector = np.zeros(self.dim)
            docs_matrix.append(doc_vector)
        X_word2vec = np.array(docs_matrix)
        if self.bow_vectorizer is None:
            return docs_matrix
        else:
            X_transformed = self.bow_vectorizer.transform(X)
            X_transformed = X_transformed.astype(np.float32)
            X_word2vec = csr_matrix(X_word2vec)
            return hstack((X_transformed, X_word2vec), format='csr')
