import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer      #countVectore
from sklearn.feature_extraction.text import TfidfVectorizer      #TF-IDF
from sklearn.model_selection import train_test_split            #train test split
from sklearn.naive_bayes import MultinomialNB                    #naive bayes
from sklearn import metrics                                     #accuracy and cunfussion matrix
import pymongo
from pymongo import MongoClient
from sklearn.externals import joblib
from config import PICKLE_FILE_PATH

def naiveBayes_model(dfx,dfy,botId,userID):
    cv=CountVectorizer(ngram_range=(1,2),max_features=1500)
    x_traincv=cv.fit_transform(dfx)
    filename = PICKLE_FILE_PATH + botId + '_nb.pkl'
    joblib.dump(cv.vocabulary_, open(filename, 'wb'))
    x_testcv=cv.transform(dfx)
    nb=MultinomialNB(alpha=0.2)
    nb.fit(x_traincv,dfy)
    pred_prob = nb.predict_proba(x_testcv)
    pred=nb.predict(x_testcv)
    filename = PICKLE_FILE_PATH + botId  + '_nb.sav'
    joblib.dump(nb, open(filename, 'wb'))

    
    
    
    
def svm_model(dfx,dfy,botId,userID):

    cv=CountVectorizer(ngram_range=(1,2))  # create object of counter vector
    #print(cv)
    x_train_cv = cv.fit_transform(dfx)
    x_test_cv = cv.transform(dfx)

    filename1= PICKLE_FILE_PATH + botId + '_svm.pkl'
    joblib.dump(cv.vocabulary_, open(filename1, 'wb'))
    table = pd.DataFrame(x_train_cv.toarray(),columns=cv.get_feature_names())
    nb = svm.SVC( kernel="linear",probability=True)
    #fit model
    nb.fit(x_train_cv, dfy)
    pred=nb.predict(x_test_cv)
    #metrics.accuracy_score(dfy, pred)
   # save the model to disk
    filename = PICKLE_FILE_PATH + botId + '_svm.sav'
    joblib.dump(nb, open(filename, 'wb'))
