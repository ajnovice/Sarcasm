import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import scikitplot as skplt
import matplotlib.pyplot as plt


def sarcasm(file_path, max_features):
    df = pd.read_csv(file_path)
    # print(list(df))
    # print(df.count())
    df.dropna(subset=['comment'], inplace=True)
    # print(df.count())
    train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
    features_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    print(features_comment)
    logit = MultinomialNB()
    pipeline = Pipeline([('features_comment', features_comment), ('logit', logit)])
    pipeline.fit(train_texts, y_train)
    print("Training Accuracy for MNB without stopwords : ", pipeline.score(train_texts, y_train) * 100)
    prediction = pipeline.predict(valid_texts)
    print("Test Accuracy for MNB without stopwords : ", accuracy_score(y_valid, prediction) * 100)
    skplt.metrics.plot_confusion_matrix(y_valid, prediction, normalize=True)
    plt.show()
    skplt.metrics.plot_roc(y_valid, np.column_stack((y_valid, prediction)), plot_micro=False, plot_macro=False)
    plt.show()


if __name__ == '__main__':
    file_path = str(sys.argv[1])
    max_features = int(sys.argv[2])
    sarcasm(file_path, max_features)
