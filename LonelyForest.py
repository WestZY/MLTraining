# coding: utf-8

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import Imputer, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import chi2,SelectPercentile
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn import metrics


class LonelyForest:
    def loadData(self, filename):
        data = pd.read_csv(filename)
        return data.ix[:, 1:], data.ix[:, 0]

    def dropColumn(self, x, feature_cols):
        return x.drop(feature_cols, axis=1)

    def sepData(self, x):
        return x.ix[:, 0:5], x.ix[:, 5:]

    def setDiscreteData(self, x, special_column_number):
        x_left = x.ix[:, 0:special_column_number] #IBK_NO
        x_right = x.ix[:, (special_column_number+1):] #APCODE...
        x = pd.Series([re.sub('\w', '10', sentence, 1) \
                       for sentence in x.ix[:, special_column_number]])
        x = pd.concat([x_left, x, x_right], axis=1)
        oneHotEncoder = OneHotEncoder(sparse=False)
        x = oneHotEncoder.fit_transform(x)
        return x

    def setLabelToMean(self, x):
        imputer = Imputer(strategy="mean", axis=1)
        x = imputer.fit_transform(x)
        return x

    def setMinMaxScaler(self, x):
        mms = MinMaxScaler()
        x = mms.fit_transform(x)
        return x

    def SelectBestAndPCA(self, x, y, percentile=90):
        #sp = SelectPercentile(chi2, percentile=percentile)
        #x = sp.fit_transform(x, y)
        pca = PCA(n_components='mle')
        x = pca.fit_transform(x)
        return x

    def sepTrainTestData(self, x, y, test_size=0.3):
        y = y.reshape(len(y), 1)
        position = np.where(y[:, 0] == -1)
        x_outlier = x[position]
        y_outlier = y[position]
        print y_outlier.shape
        x = np.delete(x, position, 0)
        y = np.delete(y, position, 0)

        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=test_size, random_state=0)

        return x_train, np.concatenate((x_test, x_outlier), axis=0), \
               y_train, np.concatenate((y_test, y_outlier), axis=0)


    def train(self, x_train, x_test):
        isolationF = IsolationForest()
        isolationF.fit(x_train)
        Y_pred_test = isolationF.predict(x_test)
        return Y_pred_test



if __name__ == "__main__":
    lf = LonelyForest()
    x, y = lf.loadData('./BANCS_OCRM/data_40303.csv')
    #y = y.reshape(len(y), 1)
    #print x.isnull().sum()
    x = lf.dropColumn(x, ['OCCUPATION', 'DEGREE0', 'CREDIT_12MONTH_CONSUME', 'METAL_AMT'])
    print x.shape,y.shape

    x_discrete, x_continous = lf.sepData(x)
    print x_discrete.shape, x_continous.shape

    x_discrete = lf.setDiscreteData(x_discrete, 1)
    print x_discrete.shape, x_continous.shape

    x = np.column_stack((x_discrete, x_continous))
    np.where(x == -1, 0, x)
    print x.shape

    x = lf.setLabelToMean(x)
    print x.shape

    x = lf.setMinMaxScaler(x)
    print x.shape

    x = lf.SelectBestAndPCA(x, y, percentile=90)
    print x.shape

    #lf.sepTrainTestData(x, y, test_size=0.25)
    x_train, x_test, y_train, y_test = lf.sepTrainTestData(x, y, test_size=0.3)
    print x_train.shape, x_test.shape, y_train.shape, y_test.shape
    y_pred = lf.train(x_train, x_test)
    #print np.array(np.where(y_pred == -1))
    #print np.arraylen(np.where(y_test == -1))
    print metrics.f1_score(y_test, y_pred, average='weighted')

