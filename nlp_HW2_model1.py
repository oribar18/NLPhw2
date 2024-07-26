import warnings
import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from gensim import downloader

warnings.filterwarnings("ignore", message="Unlike other reduction functions")

# Load pre-trained GloVe vectors
PATH = 'word2vec-google-news-300'
w2v = downloader.load(PATH)


class Vectorizer:
    def __init__(self, model):
        self.model = model

    def transform(self, X, word_to_label, dim):
        words_vectors, labels = [], []
        for word in X:
            if word not in self.model.vocab: 
                word_vec = np.zeros(dim)
            else:
                word_vec = self.model[word]
            words_vectors.append(word_vec)
            labels.append(word_to_label[word])
        return words_vectors, labels


def read_data(filepath):
    words = []
    labels = {}
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(r'\ufeff', '', line)
            if line == '\t\n' or line == '\n':
                continue
            word, label = line.split('\t')
            if label[:-1] == 'O':
                label = 0
            else:
                label = 1
            words.append(word)
            labels[word] = label
    return words, labels

# Prepare training and test data
X_train, y_train_dict = read_data('data/train.tagged')
X_test, y_test_dict = read_data('data/dev.tagged')

# Initialize vectorizer
vectorizer = Vectorizer(w2v)
dim = int(re.findall(r'\d+', PATH)[-1])
# Transform the training and test data
X_train_transformed, y_train = vectorizer.transform(set(X_train), y_train_dict, dim)
X_test_transformed, y_test = vectorizer.transform(set(X_test), y_test_dict, dim)

# Initialize and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train_transformed, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test_transformed)
# print('y_test: ', y_test)
# print('y_predict: ', list(y_pred))

# Calculate and print the F1 score
f1 = f1_score(y_test, y_pred, pos_label=1)
print(f"F1Score: {f1}")
