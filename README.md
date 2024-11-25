# Text Preprocessing and Information Retrieval Techniques

This repository demonstrates various text processing and information retrieval techniques implemented in Python, including preprocessing, indexing models, and classification.

## Table of Contents
1. [Preprocessing](#1-preprocessing)
2. [Sort-based Indexing](#2-sort-based-indexing)
3. [Single-Pass In-Memory Indexing (SPIMI)](#3-single-pass-in-memory-indexing-spimi)
4. [Logarithmic Indexing](#4-logarithmic-indexing)
5. [Binary Independence Model](#5-binary-independence-model)
6. [Vector Space Model](#6-vector-space-model)
7. [Support Vector Machine](#7-support-vector-machine)

---

## 1. Preprocessing
### Steps:
- Tokenization
- Stopword Removal
- Stemming (using Porter Stemmer)

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
    return [porter.stem(token) for token in tokens if token.isalnum() and token not in stop_words]

def process_files_in_directory(directory_path):
    processed_data = {}
    for filename in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                processed_data[filename] = preprocess_text(text)
    return processed_data
```

---

## 2. Sort-based Indexing
### Features:
- Tokenizes documents.
- Constructs an inverted index with term-document mappings.

```python
from collections import defaultdict

class SortBasedIndexing:
    # Implementation details
    ...
```

---

## 3. Single-Pass In-Memory Indexing (SPIMI)
### Features:
- Builds an inverted index in memory.
- Writes the index to a file for scalability.

```python
class InMemoryIndexer:
    # Implementation details
    ...
```

---

## 4. Logarithmic Indexing
### Features:
- Logarithmic dynamic indexing for scalable indexing.
- Supports merging and dynamic allocation of tokens.

```python
class LogarithmicDynamicIndexer:
    # Implementation details
    ...
```

---

## 5. Binary Independence Model
### Features:
- Computes term probabilities for relevance.
- Scores documents using a probabilistic approach.

```python
def preprocess(text):
    return text.lower().split()

def build_index(paths):
    # Implementation details
    ...
```

---

## 6. Vector Space Model
### Features:
- Constructs a TF-IDF matrix.
- Computes cosine similarity for ranking documents.

```python
def compute_tf_idf(index, raw_tf, num_docs):
    # Implementation details
    ...
```

---

## 7. Support Vector Machine (SVM) Classifier
### Features:
- Text classification using SVM.
- TF-IDF feature extraction.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def load_data(folder):
    # Implementation details
    ...
```

---

## Usage
1. Clone the repository.
2. Replace placeholder paths (e.g., `/path/to/documents`) with actual directories.
3. Run the scripts in your Python environment.

---

Feel free to contribute or suggest improvements! 😊
