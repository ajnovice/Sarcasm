import tensorflow.contrib.learn as skflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

file_path = 'data/train-balanced-sarcasm.csv'
df = pd.read_csv(file_path)
# print(list(df))
# print(df.count())
df.dropna(subset=['comment'], inplace=True)
# print(df.count())
train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(train_texts, y_train)
score = accuracy_score(y_valid, classifier.predict(valid_texts))
print("Accuracy: %f" % score)