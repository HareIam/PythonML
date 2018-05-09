#!/usr/bin/env python3.x
#title           :Classifier_with_feature_selection.py
#description     :This will automatic do the LOO CV classification task with Pipeline feature selection method.
#author          :XU SHIHAO
#date            :20180508
#version         :0.1
#usage           :python Classifier_with_feature_selection.py
#notes           :Input file format should be same as example
#python_version  :3.X
#==============================================================================

import csv,os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from Resampling import Resampling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from pathlib import Path

# load data
def open_file(file_name):

    ini=0
    ClassLabel=[]
    Data=[]
    with open(file_name, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # first line is feature names, last column is label
        for row in spamreader:
            if ini==0:
                FeatureNames = row[1:-1]
            else:
                ClassLabel.append(row[-1])
                Data.append([float(value) for value in row[1:-1]])
            ini=ini+1

    # Identify non-numeric label
    if ~ClassLabel[0].isdigit():
        le = preprocessing.LabelEncoder()
        le.fit(ClassLabel)
        ClassLabel=le.transform(ClassLabel)

    # return numpy format
    X_tensor=np.array(Data)
    Y_tensor=np.array(ClassLabel)
    return X_tensor,Y_tensor,FeatureNames

def Mine_Pipline(X_tensor, Y_tensor,FeatureNames,eval_model, select_model, oversampling=False):

    try:
        clf_fs = SelectFromModel(get_model(eval_model)) # feature selection
        clf_classifier=get_model(select_model)  #classifier model

        # initial parameters
        accuracy_train = 0
        accuracy_test = 0
        loo = LeaveOneOut()
        predict_label=[]
        Important_features=[]

        for train_index, test_index in loo.split(X_tensor):

            # separate training data and testing data
            X_train, X_test = X_tensor[train_index], X_tensor[test_index]
            y_train, y_test = Y_tensor[train_index], Y_tensor[test_index]

            # fit the feature selection model
            clf_fs.fit(X_train, y_train)
            X_train_fs=clf_fs.transform(X_train)

            # print ("X_train: ", X_train.shape)
            # print("X_train: ", X_train_fs.shape)

            # get selected feature and save selected features name in a list
            feature_select_bool=clf_fs.get_support()
            selected_feature=[]
            i=0
            for item in feature_select_bool:
                if item==True:
                    selected_feature.append(FeatureNames[i])
                i=i+1
            Important_features.append(selected_feature)

            # Over-sample the data after feature selection
            if (oversampling == True):
                re = Resampling()
                X_train_fs, y_train = re.smoteOversampling(X_train_fs, y_train)

            # Transform the testing data
            X_test=clf_fs.transform(X_test)

            # fit the classifier
            clf_classifier.fit(X_train_fs,y_train)

            # predict training data and testing data
            #pred_train = clf_classifier.predict(X_train_fs)
            pred = clf_classifier.predict(X_test)
            #accuracy_train += accuracy_score(y_train, pred_train)
            accuracy_test += accuracy_score(y_test, pred)
            # print(y_test,pred)
            # print(accuracy_test)
            predict_label.append(pred)

        acc_result=(accuracy_test / len(X_tensor))
        # print("Classifier: ",select_model, "Evaluator: ",eval_model)
        # print("Gensim accuracy_train:", accuracy_train / len(X_tensor))
        # print("Gensim accuracy_test:", accuracy_test / len(X_tensor))
        predict_label=np.array(predict_label)
        fpr, tpr, thresholds = metrics.roc_curve(Y_tensor, predict_label, pos_label=1)
        auc1=metrics.auc(fpr, tpr)
        CM=confusion_matrix(Y_tensor, predict_label, labels=[1,0]).ravel()
        CR=classification_report(Y_tensor, predict_label)
        # print("Confusion Matrix: ",CM)
        #print(CR,"AUC= ",auc1)
        return acc_result,CM,CR,auc1,predict_label,Important_features
    except:
        return -1,-1,-1,-1,-1,-1

def get_model(name):
    if name == 'LogisticRegression':
        return LogisticRegression()
    elif name == 'SVM_linear':
        return LinearSVC()
    elif name == 'Decision_tree':
        return tree.DecisionTreeClassifier()
    elif name == 'SVM_poly':
        return SVC(kernel = 'poly')
    elif name == 'SVM_rbf':
        return SVC(kernel = 'rbf')
    elif name == 'MultinomialNB':
        return MultinomialNB()
    elif name == 'GradientBoostingClassifier':
        return GradientBoostingClassifier(n_estimators=200)
    elif name == 'KNeighborsClassifier':
        return KNeighborsClassifier(1)
    elif name == 'MLPClassifier':
        return MLPClassifier(alpha=1)
    elif name == 'NaiveBayes':
        return GaussianNB()

    else:
        raise ValueError('No such model')

def save_result(save_name,result_great,acc_max,AUC_max,ConfuMatrix_max,ClfReport_max,Important_features):
    file = open(save_name, 'a')
    file.write("-------Classifier: ")
    file.write(result_great[1])
    file.write("-------Evaluator: ")
    file.write(result_great[2])
    file.write("--------\n")
    file.write("Acc=")
    file.write(str(acc_max))
    file.write("AUC: ")
    file.write(str(AUC_max))
    file.write("\n")
    file.write("Confusion Matrix: \n")
    file.write(str(ConfuMatrix_max))
    file.write("\n")
    file.write("Classification report: ")
    file.write(str(ClfReport_max))
    file.write("\n")
    for item in Important_features:
        file.write(str(item))
        file.write("\n")
    file.close()

if __name__ == '__main__':

    file_name_pool=os.listdir('./data')
    SMOTE_=False
    for file_name in file_name_pool:
        print("----------"+file_name+'---------------')
        File_attress= './data/'+file_name
        if(SMOTE_==True):
            save_name = './result/result_SMOTE' + file_name[0:-4] + '.txt'
        else:
            save_name = './result/result_' + file_name[0:-4] + '.txt'

        my_file = Path(save_name)
        if not my_file.is_file():
            # open csv file
            Data,lable,FeatureNames=open_file(File_attress)

            # normalize data
            #Data = preprocessing.normalize(Data)
            Data = preprocessing.scale(Data)

            Classifier_list=['LogisticRegression',
                             'SVM_linear',
                             'GradientBoostingClassifier']
                             # 'Decision_tree',
                             # 'SVM_poly',
                             # 'SVM_rbf',
                             # 'MultinomialNB',
                             # 'KNeighborsClassifier',
                             # 'NaiveBayes']
            result_buff=[]
            acc_max=-2
            for Classifier in Classifier_list:
                for Evaluator in Classifier_list:
                    clf_acc_, ConfuMatrix_, ClfReport_, AUC_, predict_label_,Important_features = Mine_Pipline(Data,lable, \
                                            FeatureNames,Evaluator,Classifier,oversampling=SMOTE_)
                    if clf_acc_!=-1:
                        if clf_acc_ > acc_max:
                            # save maximum settings
                            acc_max=clf_acc_
                            ConfuMatrix_max=ConfuMatrix_
                            ClfReport_max=ClfReport_
                            AUC_max=AUC_
                            predict_label_max=predict_label_
                        result_buff.append([clf_acc_, Classifier, Evaluator])
                    else:
                        print("I am in error")
                        continue
                    print([clf_acc_, Classifier, Evaluator])

            if acc_max!=-2:
                result_great=max(result_buff)
                print("-------Classifier: ", result_great[1], "-------Evaluator: ", result_great[2], "--------")
                print("Acc=", acc_max, "AUC: ", AUC_max)
                print("Confusion Matrix: \n", ConfuMatrix_max)
                print("Classification report: ", ClfReport_max)
                save_result(save_name, result_great, acc_max, AUC_max,ConfuMatrix_max, ClfReport_max, Important_features)








