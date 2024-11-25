# Text Processing and Information Retrieval Models

## Table of Contents
1. [Preprocessing: Stopword Removal and Normalization](#1-preprocessing-stopword-removal-and-normalization)
2. [Sort-Based Indexing](#2-sort-based-indexing)
3. [SPIMI (Single-Pass In-Memory Indexing)](#3-spimi-single-pass-in-memory-indexing)
4. [Logarithmic Indexing](#4-logarithmic-indexing)
5. [Binary Independence Model (BIM)](#5-binary-independence-model-bim)
6. [Vector Space Model (VSM)](#6-vector-space-model-vsm)
7. [Support Vector Machine (SVM) Classifier for Text Data](#7-support-vector-machine-svm-classifier-for-text-data)
8. [Conclusion](#conclusion)

This repository contains Python implementations of various text processing and information retrieval models. The code is divided into multiple sections, each addressing a specific technique or model.

---

## 1. Preprocessing: Stopword Removal and Normalization

```python
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    processed_tokens = [porter.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    
    return processed_tokens

def process_files_in_directory(directory_path):
    processed_data = {}
    
    for filename in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                processed_data[filename] = preprocess_text(text)
    
    return processed_data

directory_path = "/content/drive/MyDrive/YourFolderName"  
processed_files = process_files_in_directory(directory_path)
for filename, tokens in processed_files.items():
    print(f"{filename}: {tokens[:10]}...") 
```

---

## 2. Sort-Based Indexing

```python
import os
import re
from collections import defaultdict

class SortBasedIndexing:
    def __init__(self):
        self.term_document_map = defaultdict(list)
        self.documents = []

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, doc_id, text):
        tokens = self.tokenize(text)
        self.documents.append((doc_id, tokens))
        for position, term in enumerate(tokens):
            self.term_document_map[term].append((doc_id, position))

    def sort_terms(self):
        sorted_terms = sorted(self.term_document_map.items())
        return sorted_terms

    def create_inverted_index(self):
        inverted_index = defaultdict(list)
        sorted_terms = self.sort_terms()
        for term, postings in sorted_terms:
            inverted_index[term] = sorted(postings)
        return inverted_index

    def display_inverted_index(self, inverted_index):
        for term, postings in inverted_index.items():
            doc_ids = [doc_id for doc_id, _ in postings]
            print(f"{term}: {doc_ids}")

documents_path = "/content/drive/MyDrive/Colab Notebooks/Final Data"

indexing = SortBasedIndexing()

for doc_id, file_name in enumerate(os.listdir(documents_path)):
    file_path = os.path.join(documents_path, file_name)
    if os.path.isfile(file_path): 
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            indexing.add_document(doc_id, content)

inverted_index = indexing.create_inverted_index()
indexing.display_inverted_index(inverted_index)
```

---

## 3. SPIMI (Single-Pass In-Memory Indexing)

```python
import nltk
from google.colab import drive
import os
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')

class InMemoryIndexer:
    def __init__(self, input_dir, output_file):
        self.input_dir = input_dir
        self.output_file = output_file
        self.index = defaultdict(list)
        self.doc_map = {}
        self.porter = PorterStemmer()

    def build_index(self):
        for doc_id, filename in enumerate(sorted(os.listdir(self.input_dir))):
            path = os.path.join(self.input_dir, filename)
            self.doc_map[doc_id] = filename
            with open(path, 'r', encoding='utf-8') as f:
                tokens = {self.porter.stem(token) for token in word_tokenize(f.read().lower())}
                for token in tokens:
                    self.index[token].append(doc_id)
        self.write_index()

    def write_index(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for term, doc_ids in sorted(self.index.items()):
                f.write(f"{term} ({len(doc_ids)}) - [{','.join(map(str, sorted(doc_ids)))}]\n")

    def print_doc_map(self):
        for doc_id, filename in self.doc_map.items():
            print(f"Document ID: {doc_id}, Filename: {filename}")

input_dir = "/content/drive/MyDrive/Colab Notebooks/Final Data"
output_file = "/content/drive/MyDrive/Colab Notebooks/single_pass_output.txt"

indexer = InMemoryIndexer(input_dir, output_file)
indexer.build_index()
indexer.print_doc_map()
```

---

## 4. Logarithmic Indexing

```python
import os
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')

class LogarithmicDynamicIndexer:
    def __init__(self, output_file, n_indexes=2, n=10):
        self.n_indexes = n_indexes  
        self.n = n  
        self.indexes = self.create_indexes(n_indexes, n)  
        self.filled = []  
        self.z = []
        self.porter = PorterStemmer()
        self.output_file = output_file

    def create_indexes(self, n_indexes, n):
        
        return [[] for _ in range(n_indexes)]

    def tokenize_and_process(self, text):
       
        tokens = word_tokenize(text.lower())
        processed_tokens = [self.porter.stem(token) for token in tokens]
        return [token for token in processed_tokens if token.isalpha()]  

    def merge_add_token(self, token):
        
        self.z.append(token)
        if len(self.z) == self.n:
            self._merge_levels(0)  
            self.z = []  #

    def _merge_levels(self, level):
    
        if level >= self.n_indexes:
            return

        if self.indexes[level] in self.filled:
            self.z += self.indexes[level]
            self.filled.remove(self.indexes[level])

            self.indexes[level] = []
            self._merge_levels(level + 1)
        else:
            self.indexes[level] = self.z
            self.filled.append(self.indexes[level])
    def flush_buffer(self):
        if self.z:
            print(f"Flushing buffer: {self.z}")
            self._merge_levels(0)
            self.z = []
            
```

**(Continued in next part)**

```markdown
---

## 5. Binary Independence Model (BIM)

```python
from collections import defaultdict
import math

class BinaryIndependenceModel:
    def __init__(self):
        self.doc_terms = defaultdict(list)  # Maps document IDs to terms
        self.term_doc_count = defaultdict(int)  # Tracks how many documents contain each term
        self.num_docs = 0  # Total number of documents

    def add_document(self, doc_id, terms):
        self.doc_terms[doc_id] = terms
        unique_terms = set(terms)
        for term in unique_terms:
            self.term_doc_count[term] += 1
        self.num_docs += 1

    def calculate_probabilities(self, query_terms):
        scores = defaultdict(float)
        for term in query_terms:
            if term in self.term_doc_count:
                prob_term_in_relevant = self.term_doc_count[term] / self.num_docs
                prob_term_not_in_relevant = 1 - prob_term_in_relevant
                for doc_id, doc_terms in self.doc_terms.items():
                    if term in doc_terms:
                        scores[doc_id] += math.log(prob_term_in_relevant / prob_term_not_in_relevant)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

bim = BinaryIndependenceModel()
documents = [
    (1, ['cat', 'dog', 'fish']),
    (2, ['cat', 'lion', 'zebra']),
    (3, ['dog', 'wolf']),
    (4, ['dog', 'cat', 'fish']),
]

for doc_id, terms in documents:
    bim.add_document(doc_id, terms)

query = ['cat', 'dog']
results = bim.calculate_probabilities(query)
print("Ranked Results:", results)
```

---

## 6. Vector Space Model (VSM)

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorSpaceModel:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.corpus = corpus

    def query(self, q):
        query_vector = self.vectorizer.transform([q])
        cosine_similarities = np.dot(self.tfidf_matrix, query_vector.T).toarray().flatten()
        return np.argsort(cosine_similarities)[::-1]

corpus = [
    "cat and dog",
    "dog and wolf",
    "cat and fish",
    "lion and tiger",
]

vsm = VectorSpaceModel(corpus)
query = "cat and dog"
results = vsm.query(query)
print("Ranked Document Indices:", results)
```

---

## 7. Support Vector Machine (SVM) Classifier for Text Data

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example data
corpus = [
    "cat dog lion",
    "dog wolf tiger",
    "cat tiger lion",
    "wolf dog fish",
]
labels = [0, 1, 0, 1]  # 0: Animals, 1: Wild Animals

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train SVM Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Predictions
predictions = svm_classifier.predict(X_test_tfidf)
print(classification_report(y_test, predictions))
```

---

### Conclusion

This repository covers fundamental and advanced indexing and retrieval techniques, ensuring a comprehensive guide for working with text data. Each section includes implementation and detailed explanation. Contributions are welcome!

---
```

Feel free to copy and modify this content for your project! It now includes all the sections comprehensively.
