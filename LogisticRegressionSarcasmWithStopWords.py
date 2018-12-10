import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# file_path = 'data/train-balanced-sarcasm.csv'


def sarcasm(file_path, max_features, solver):
    df = pd.read_csv(file_path)
    # print(list(df))
    # print(df.count())
    df.dropna(subset=['comment'], inplace=True)
    # print(df.count())
    train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
    features_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words='english')
    logit = LogisticRegression(solver=solver, random_state=17, max_iter=5000)
    pipeline = Pipeline([('features_comment', features_comment), ('logit', logit)])
    pipeline.fit(train_texts, y_train)
    print("Training Accuracy for Logistic Regression with stopwords : ", pipeline.score(train_texts, y_train) * 100)
    prediction = pipeline.predict(valid_texts)
    print("Test Accuracy for Logistic Regression with stopwords : ", accuracy_score(y_valid, prediction) * 100)


if __name__ == '__main__':
    file_path = str(sys.argv[1])
    max_features = int(sys.argv[2])
    solver = str(sys.argv[3])
    sarcasm(file_path, max_features, solver)
