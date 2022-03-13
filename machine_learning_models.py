#-*- coding: UTF-8 -*-

import warnings
warnings.filterwarnings("ignore")


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import tree
from sklearn import svm
import xgboost as xgb


from evaluation import svmEvaluate
from evaluation import evaluate

#from drawpic import crossrocauc
#from drawpic import rocauc

def logisticregression(trainsdata,traintags,testsdata,testtags):
    print("Logistic Regression")
    model = linear_model.LogisticRegression()
    print("Training")
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def bayes(trainsdata, traintags,testsdata,testtags):
    print("GaussianNB")
    by = GaussianNB()
    print("Training")
    by.fit(trainsdata,traintags)
    y_pred = by.predict(testsdata)
    y_score = by.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def Lda(trainsdata, traintags,testsdata,testtags):
    print("LDA")
    by = LinearDiscriminantAnalysis()
    print("Training")
    by.fit(trainsdata,traintags)
    y_pred = by.predict(testsdata)
    y_score = by.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def decisiontree(trainsdata,traintags,testsdata,testtags):
    print("Decision Tree")
    model = DecisionTreeClassifier()
    print("Training ")
    model.fit(trainsdata,traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def Svm(trainsdata,traintags,testsdata,testtags):
    print ("SVM")
    params = {'C': 9.928279022403954, 'gamma': 0.00014496253219114155}
    model = SVC(C=params['C'], gamma=params['gamma'], probability=True)
    print("Training ")
    model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    return  svmEvaluate(testtags, y_pred,y_score)

def svmOpt(testsdata,testtags,model):
    print ("SvmOpt Classifier")
#    from sklearn import svm
#    model = svm.SVC(probability=True)
#    print("Svm training")
#    model.fit(trainsdata, traintags)
    y_pred = model.predict(testsdata)
    y_score = model.predict_proba(testsdata)
    # print y_score
#    print y_pred
#    print y_score
#    y_score = y_pred
#    for i in range(len(y_pred)):
#        if y_pred[i]>0.5:y_pred[i]=1
#        else:y_pred[i]=0
    from evaluation import svmEvaluate
    return  svmEvaluate(testtags, y_pred,y_score)

def randomforest(trainsdata,traintags,testsdata,testtags):
    print ("Random Forest")
    rf0 = RandomForestClassifier(oob_score=True)
    print ("Training")
    rf0.fit(trainsdata,traintags)
    y_pred = rf0.predict(testsdata)
    y_score = rf0.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def adaboost(trainsdata,traintags,testsdata,testtags):
    print("AdaBoost")
    clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    print("Training")
    clf.fit(trainsdata,traintags)
    y_pred = clf.predict(testsdata)
    y_score = clf.predict_proba(testsdata)
    return evaluate(testtags, y_pred,y_score)

def XGBoost(traindata,traintags,testdata,testtags):
    print("XGBoost")
    clf = xgb.XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=50,subsample=1,colsample_bytree=0.8,gamma=0.3,max_depth=3,min_child_weight=4,learning_rate=0.3)
    print("Training")
    clf.fit(traindata,traintags)
    y_pred = clf.predict(testdata)
    y_score = clf.predict_proba(testdata)
    return evaluate(testtags, y_pred,y_score)

def LightGBM(traindata,traintags,testdata,testtags):
    print("LightGBM")
    clf = LGBMClassifier()
    print("Training")
    clf.fit(traindata,traintags)
    y_pred = clf.predict(testdata)
    y_score = clf.predict_proba(testdata)
    return evaluate(testtags, y_pred,y_score)

def RANDOMTrainStag(traindata,traintags,testdata,testtags):
    clf = KNeighborsClassifier()
    clf.fit(traindata,traintags)
    train_label = clf.predict(traindata)
    y_pred = clf.predict(testdata)
    for i in range(len(train_label)):
        if train_label[i]>0.5:train_label[i]=1
        else:train_label[i]=0
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:y_pred[i]=1
        else:y_pred[i]=0
    return (train_label,y_pred)

def XGBoostTrainStag(traindata,traintags,testdata,testtags):
    train_data = xgb.DMatrix(traindata, label=traintags)
    test_data = xgb.DMatrix(testdata)
    params = {'booster': 'gbtree',
               'objective': 'binary:logistic',
               'eval_metric': 'auc',
               'gamma': 0.3,#default=0，一般取值为0-0.5
               'random_state': 50,
               'max_depth': 3,#default=6，一般取值为3-10
               'min_child_weight': 4,#default=1，一般取值为1-6
               'lambda': 1,#default=1
               'subsample': 1,#default=1，一般取值为0.5-1
               'colsample_bytree': 0.8,#default=1，一般取值为0.5-1
               'eta': 0.3,#default=0.3，一般取值为0.01-0.2,即learning_rate
               }
    clf = xgb.train(params,train_data)
    train_label = clf.predict(train_data)
    y_pred = clf.predict(test_data,validate_features=False)
    for i in range(len(train_label)):
        if train_label[i]>0.5:train_label[i]=1
        else:train_label[i]=0
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:y_pred[i]=1
        else:y_pred[i]=0
    return (train_label,y_pred)