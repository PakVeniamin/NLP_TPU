import math
import re
from collections import Counter

class TFIDF:

    def __init__(self, documents):
        self.documents = documents 
        self.vocab = self.build_vocabulary()

    def build_vocabulary(self):
        vocabulary = set()
        for document in self.documents:
            vocabulary.update(document.split())
        return vocabulary

    def term_frequency(self, document):
        words = document.split()
        word_count = Counter(words)
        total_words = len(words)
        return {word: count/total_words for word, count in word_count.items()}
    
    def inverse_document_frequency(self):
        document_count = len(self.documents)
        idf = {}
        for word in self.vocab:
            doc_words_count = sum(1 for doc in self.documents if word in doc)
            idf[word] = math.log(document_count/doc_words_count)
        return idf

    def tf_idf(self, document):
        tf  = self.term_frequency(document)
        idf = self.inverse_document_frequency()
        tfidf = {}
        for word, freq in tf.items():
            tfidf[word] = freq * idf[word]
        return tfidf
    
    def tf(self, document):
        return self.term_frequency(document)

    def idf(self):
        return self.inverse_document_frequency()
    

documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document",
]

# tfidf_calculator = TFIDF(documents)

# # Вывод TF-IDF вектора для каждого документа
# for idx, doc in enumerate(documents):
#     print("\nTF-IDF Vector for Document", idx+1)
#     tfidf_vector = tfidf_calculator.tf_idf(doc)
#     print(tfidf_vector)

# # Вывод TF для каждого документа
# for idx, doc in enumerate(documents):
#     print("\nTF for Document", idx+1)
#     tf_vector = tfidf_calculator.tf(doc)
#     print(tf_vector)

# # Вывод IDF для всего корпуса документов
# print("\nIDF for Vocabulary")
# idf_values = tfidf_calculator.idf()
# print(idf_values)
    

class BagOfWords:
    def __init__(self):
        self.vocabulary = {}

    def fit(self, documents):
        for document in documents:
            words = document.split()
            word_count = Counter(words)
            for word, count in word_count.items():
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
    
    def transform(self, documents):
        vectors = []
        for document in documents:
            vector = [0] * len(self.vocabulary)
            words = document.split()
            for word in words:
                if word in self.vocabulary:
                    index = self.vocabulary[word]
                    vector[index] += 1
            vectors.append(vector)
        return vectors
    
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document",
]

# Создаем экземпляр класса BagOfWords
bow = BagOfWords()

# Обучаем модель на документах
bow.fit(documents)

# Преобразуем документы в векторы мешка слов
bow_vectors = bow.transform(documents)

# Вывод результатов
print("Vocabulary:", bow.vocabulary)
print("\nBag of Words Vectors:")
for idx, vector in enumerate(bow_vectors):
    print("Document", idx+1, ":", vector)

"""История для Word2Vec"""

def extract_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(" ".join(words[i:i+n]))
    return ngrams

def generate_word_pairs(text, window_size):
    words = text.split()
    word_pairs = []
    for left in range (len(words)):
        for right in range(left + 1, min(left+window_size+1, len(words))):
            word_pairs.append((words[left], words[right]))
        
    return word_pairs

def build_vocab(text, vocab_type='both'):
    words = re.findall(r'[^a-z@# ]', text.lower()) 
    unique_words = set(words)

    def word_to_index():
        return {word: index for index, word in enumerate(unique_words)}
    
    def index_to_word():
        return {index: word for index, word in enumerate(unique_words)}
    
    def both():
        return word_to_index(), index_to_word()
    
    vocab_functions = {
        'word_to_index': word_to_index,
        'index_to_word': index_to_word,
        'both': both
    }

    if vocab_type in vocab_functions:
        return vocab_functions[vocab_type]()
    else:
        raise ValueError("Invalid vocab_type. Please Choose 'word_to_index', 'index_to_word' or 'both'.")