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

 

# code for predictions
    
    
import joblib
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from config import TRAINED_MODEL_PATH

def loadModels(botId,userID):
    ### Import Naive bayes model
    NB_Model = TRAINED_MODEL_PATH + botId   + '_nb.sav'
    NB_Vocab = TRAINED_MODEL_PATH + botId   + '_nb.pkl'
    ### Import SVM model
    SVM_Model = TRAINED_MODEL_PATH + botId   + '_svm.sav'
    SVM_Vocab = TRAINED_MODEL_PATH + botId   + '_svm.pkl'
    return NB_Model,NB_Vocab,SVM_Model,SVM_Vocab

def loadNB(filename, vocab):
    loaded_vec = CountVectorizer(vocabulary=joblib.load(open(vocab, "rb")))
    loaded_model = joblib.load(open(filename, 'rb'))
    return loaded_vec,loaded_model

## loading SVM vocab and model from disk
def loadSVM(filename,vocab):
    loaded_vec=CountVectorizer(vocabulary=joblib.load(open(vocab, "rb")))
    loaded_model = joblib.load(open(filename, 'rb'))
    return loaded_vec,loaded_model


def predictions(question,botId,userID):
    NB_Model,NB_Vocab,SVM_Model,SVM_Vocab = loadModels(botId,userID)
    ### loading all vocab and model
    loaded_nbVocab, loaded_nb = loadNB(NB_Model, NB_Vocab)
    loaded_svmVocab, loaded_svm = loadSVM(SVM_Model, SVM_Vocab)
    new_cv_nb = loaded_nbVocab.transform(question)
    new_cv_svm = loaded_svmVocab.transform(question)
    if new_cv_nb.getnnz() == 0 and new_cv_svm.getnnz() == 0:
        return "unidentified intent"
    else:
        nb_pred = loaded_nb.predict(new_cv_nb)
        nb_prob = loaded_nb.predict_proba(new_cv_nb)
        svm_pred = loaded_svm.predict(new_cv_svm)
        svm_prob = loaded_svm.predict_proba(new_cv_svm)
        for item in nb_prob:
            maxprob_nb = (max(item))
        for element in svm_prob:
            maxprob_svm = (max(element))
        return [nb_pred,svm_pred,maxprob_nb,maxprob_svm]

## ensemble for voting for prediction
def ensemblemethod(x,y,a,b):
    if x == y:
        return [x]
    elif x != y:
        if a >= b:
            return [x]
        else:
            return [y]


def startPrediction(message,botResponseInJson):
    #question = ['when is diwali']
    try:
        pred_Result = predictions(message,botResponseInJson['botID'],botResponseInJson['userID'])
        #print(pred_Result)
        if pred_Result == "unidentified intent":
            result = pred_Result
            return result
        else:
            NBResult = pred_Result[0][0]
            SVMResult = pred_Result[1][0]
            NBScore = pred_Result[2]
            SVMScore = pred_Result[3]
            #print(NBScore,SVMScore)
            result = ensemblemethod(NBResult,SVMResult,NBScore,SVMScore)
            #print(result)
            #print(len(result))
            return result[0]
    except:
        return 'Please Train the BOT'
