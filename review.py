import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.pipeline import Pipeline
from sklearn import metrics


yelp=pd.read_csv('review.csv')
x=yelp['text']
y=yelp['stars']

'''###BASIC TEXT PROCESSING TO TOKENIZE THE DOCUMENTS.NOT IMPLEMENTED FOR THE GIVEN DATASET
def text_process(raw_text):
    nop = [char for char in raw_text if char not in string.punctuation]
    nop = ''.join(nopunc)
    return [word for word in nop.split() if word.lower() not in stopwords.words('english')]
'''

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=101)
pipe=Pipeline([('c',CountVectorizer()),('m',MultinomialNB())])
pipe.fit(x_train,y_train)
star_pred=pipe.predict(x_test)

# sparse_x=CountVectorizer().fit_transform(x)
# # Sparse_X=TfidfTransformer().fit_transform(sparse_x)
# x_train, x_test, y_train, y_test = train_test_split(Sparse_X,y, test_size=0.15, random_state=101)
# M=MultinomialNB().fit(x_train,y_train)
#star_pred=M.predict(x_test)

print('Prediction model for all ratings')
print('No of training data points:',len(y_train))
print('length of testing data points',len(y_test))
print('Confusion Matrix:')
print(confusion_matrix(y_test,star_pred))
print('Classification matrix')
print(classification_report(y_test,star_pred))

yelp15=yelp[(yelp['stars']==1)|(yelp['stars']==5)]
X=yelp15['text']
Y=yelp15['stars']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.15, random_state=101)
pipe=Pipeline([('c',CountVectorizer()),('m',MultinomialNB())])
pipe.fit(x_train,y_train)
star_pred=pipe.predict(x_test)

print('\nPrediction model for ratings 1 and 5')
print('No of training data points:',len(y_train))
print('length of testing data points',len(y_test))
print('Confusion Matrix:')
print(confusion_matrix(y_test,star_pred))
print('Classification matrix')
print(classification_report(y_test,star_pred))