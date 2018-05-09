import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
import csv
from sklearn.feature_selection.rfe import RFE, RFECV
from sklearn.datasets import load_iris, make_friedman1
from sklearn.metrics import zero_one_loss
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.utils import check_random_state
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_greater, assert_equal, assert_true

from sklearn.metrics import make_scorer
from sklearn.metrics import get_scorer
from sklearn.linear_model import LogisticRegression

def test_rfecv(X,y):
    # generator = check_random_state(0)
    # iris = load_iris()
    # X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    # y = list(iris.target)  # regression test: list should be supported
    # print(X)
    # print(y)

    # Test using the score function
    scorer = get_scorer('accuracy')
    rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10,scoring=scorer)
    rfecv.fit(X, y)
    X_r = rfecv.transform(X)
    print(rfecv.ranking_)
    print(rfecv.support_)

    print(X_r.shape)
    # All the noisy variable were filtered out
    #assert_array_equal(X_r, X)

    # same in sparse
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=1, cv=5)
    X_sparse = sparse.csr_matrix(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    print(X_r_sparse.shape)
    #assert_array_equal(X_r_sparse.toarray(), X)

    # Test using a customized loss function
    scoring = make_scorer(zero_one_loss, greater_is_better=False)
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, cv=5,
                  scoring=scoring)
    ignore_warnings(rfecv.fit)(X, y)
    X_r = rfecv.transform(X)
    print(X_r.shape)
    #assert_array_equal(X_r, X)

    # Test using a scorer
    scorer = get_scorer('accuracy')
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, cv=5,
                  scoring=scorer)
    rfecv.fit(X, y)
    X_r = rfecv.transform(X)
    print(X_r.shape)
    #assert_array_equal(X_r, X)

    # Same as the first two tests, but with step=2
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=2, cv=5)
    rfecv.fit(X, y)
    X_r = rfecv.transform(X)
    print(X_r.shape)
    #assert_array_equal(X_r, X)

    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=2, cv=5)
    X_sparse = sparse.csr_matrix(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    print(X_r_sparse.shape)
    #assert_array_equal(X_r_sparse.toarray(), X)


def open_file(file_name):
    url = ".\\data\\doc_embeddings_0_15.csv"
    ini=0
    ClassLabel=[]
    Data=[]
    with open(file_name, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # first line is feature names, last column is label
        for row in spamreader:
            if ini==0:
                FeatureNames = row[0:-1]
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
    return(X_tensor,Y_tensor)

file_name = '.\\Kaldi_LIWC_Diction_0411.csv'
# file_name='.\\data\\SUNXIN.csv'
Data,lable=open_file(file_name)
print(Data)
print(lable)
test_rfecv(Data,list(lable))
