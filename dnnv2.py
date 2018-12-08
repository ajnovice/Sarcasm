# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:24:19 2018

@author: Abdullah
"""

#import tensorflow.contrib.learn as skflow
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def train_input_fn(features, labels):
    """An input function for training"""

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    #dataset = dataset.shuffle(10).repeat().batch(batch_size)
     
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset

tf.reset_default_graph()
file_path = 'sarcasm/train-balanced-sarcasm.csv'
df = pd.read_csv(file_path)
# print(list(df))
# print(df.count())
df.dropna(subset=['comment'], inplace=True)
# print(df.count())
train_texts, valid_texts, y_train, y_valid = train_test_split(df['comment'], df['label'], test_size = 0.3, random_state=17)
#classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
#classifier = tf.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
feature_columns = []
#print(train_texts)
#print(valid_texts)
#print(y_train)
#print(y_valid)
for key in train_texts.keys():
    #print(key)
    feature_columns.append(tf.feature_column.numeric_column(key=str(key)))
print(feature_columns)    

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 10],n_classes=2)
#classifier.fit(train_texts, y_train)
batch_size = 100
train_steps = 400

for i in range(0,100): 
    classifier.train(
        input_fn=lambda:train_input_fn(train_texts,y_train),
        steps=train_steps)

predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(valid_texts,labels=None,
    batch_size=batch_size))

results = list(predictions)
print(predictions)
#score = accuracy_score(y_valid, classifier.predict(valid_texts))
#print("Accuracy: %f" % score)
