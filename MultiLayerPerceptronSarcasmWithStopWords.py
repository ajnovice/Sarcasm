import sys
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

file_path = 'data/train-balanced-sarcasm.csv'
df = pd.read_csv(file_path)
# print(list(df))
# print(df.count())
df.dropna(subset=['comment'], inplace=True)
# print(df.count())
train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
features_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=500000, min_df=2)
print(features_comment)
logit = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=17)
pipeline = Pipeline([('features_comment', features_comment), ('logit', logit)])
pipeline.fit(train_texts, y_train)
prediction = pipeline.predict(valid_texts)
skplt.metrics.plot_confusion_matrix(y_valid, prediction, normalize=True)
plt.show()
skplt.metrics.plot_roc(y_valid, np.column_stack((y_valid, prediction)), plot_micro=False, plot_macro=False)
plt.show()
print("Training Accuracy for MLP without stopwords : ", pipeline.score(train_texts, y_train) * 100)
prediction = pipeline.predict(valid_texts)
print("Test Accuracy for MLP without stopwords : ", accuracy_score(y_valid, prediction) * 100)

# def sarcasm(file_path, max_features, solver, max_items):
#     df = pd.read_csv(file_path)
#     df = df[0:max_items]
#     # print(list(df))
#     # print(df.count())
#     df.dropna(subset=['comment'], inplace=True)
#     # print(df.count())
#     train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
#     features_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
#     logit = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=17)
#     pipeline = Pipeline([('features_comment', features_comment), ('logit', logit)])
#     pipeline.fit(train_texts, y_train)
#     print("Training Accuracy for SVM with stopwords : ", pipeline.score(train_texts, y_train) * 100)
#     prediction = pipeline.predict(valid_texts)
#     print("Test Accuracy for SVM with stopwords : ", accuracy_score(y_valid, prediction) * 100)
#
#
# if __name__ == '__main__':
#     file_path = str(sys.argv[1])
#     max_features = int(sys.argv[2])
#     solver = str(sys.argv[3])
#     max_items = int(sys.argv[4])
#     sarcasm(file_path, max_features, solver, max_items)