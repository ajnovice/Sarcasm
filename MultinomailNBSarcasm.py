import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

file_path = 'data/train-balanced-sarcasm.csv'
df = pd.read_csv(file_path)
# print(list(df))
# print(df.count())
df.dropna(subset=['comment'], inplace=True)
# print(df.count())
train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], random_state=17)
features_comment = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
print(features_comment)
logit = MultinomialNB()
pipeline = Pipeline([('features_comment', features_comment), ('logit', logit)])
pipeline.fit(train_texts, y_train)
prediction = pipeline.predict(valid_texts)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(accuracy_score(y_valid, prediction))
