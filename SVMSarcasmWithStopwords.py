import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def sarcasm(file_path, max_features, kernel, max_items):
    df = pd.read_csv(file_path)
    df = df[0:max_items]
    # print(list(df))
    # print(df.count())
    df.dropna(subset=['comment'], inplace=True)
    # print(df.count())
    train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
    features_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    logit = SVC(kernel=kernel, gamma='scale')
    pipeline = Pipeline([('features_comment', features_comment), ('logit', logit)])
    pipeline.fit(train_texts, y_train)
    print("Training Accuracy for SVM with stopwords : ", pipeline.score(train_texts, y_train) * 100)
    prediction = pipeline.predict(valid_texts)
    print("Test Accuracy for SVM with stopwords : ", accuracy_score(y_valid, prediction) * 100)


if __name__ == '__main__':
    file_path = str(sys.argv[1])
    max_features = int(sys.argv[2])
    kernel = str(sys.argv[3])
    max_items = int(sys.argv[4])
    sarcasm(file_path, max_features, kernel, max_items)
