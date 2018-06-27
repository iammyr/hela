import preprocessing as pre
import pandas as pd
from const import CONST

X_test = pd.read_csv(CONST.X_TEST)
Y_train = pd.read_csv(CONST.Y_TRAIN)
Y_test = pd.read_csv(CONST.Y_TEST)
X_train = pd.read_csv(CONST.X_TRAIN)
# feature scaling, i.e., limit the range of variables 
# so that they can be compared on common grounds
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])

print 'KNN ACCURACY SCORE:'
model = pre.train_knn(X_train[CONST.COLS], Y_train.values.ravel(), CONST.COLS)
print pre.get_accuracy(X_test[CONST.COLS], Y_test, model)
print 'Y_train.Target.value_counts()' + str(Y_train.Target.value_counts())
print 'Y_train.Target.count()' + str(Y_train.Target.count())
print Y_train.Target.value_counts()/Y_train.Target.count()
print 'ACCURACY JUST BY GUESSING'
print 'Y_test.Target.value_counts()' + str(Y_test.Target.value_counts())
print 'Y_test.Target.count()' + str(Y_test.Target.count())
print Y_test.Target.value_counts()/Y_test.Target.count()

X_train_minmax, X_test_minmax = pre.scaling(X_train[CONST.COLS], X_test[CONST.COLS])
print 'KNN ACCURACY after scaling'
model = pre.train_knn(X_train_minmax, Y_train.values.ravel(), CONST.COLS)
print pre.get_accuracy(X_test_minmax, Y_test, model) 

print 'LINEAR REGRESSION'
encoded_y_train, encoded_y_test = pre.encoding_categorical(Y_train, Y_test)
model = pre.train_linear_regression(X_train[CONST.COLS], encoded_y_train)
print 'ACCURACY before scaling'
print pre.get_accuracy_rounded(X_test[CONST.COLS], encoded_y_test, model)
print 'ACCURACY after scaling'
print pre.get_accuracy_rounded(X_test_minmax, encoded_y_test, model)

print 'LOGISTIC REGRESSION'
model = pre.train_logistic_regression(X_train[CONST.COLS], encoded_y_train.values.ravel())
print 'ACCURACY before scaling'
print pre.get_accuracy_rounded(X_test[CONST.COLS], encoded_y_test, model)
print 'ACCURACY after scaling'
print pre.get_accuracy_rounded(X_test_minmax, encoded_y_test, model)
X_train_enc_scale, X_test_enc_scale = pre.scaling(X_train[CONST.COLS], X_test[CONST.COLS])
model = pre.train_logistic_regression(X_train_enc_scale, Y_train.values.ravel())
print 'ACCURACY after standardising'
print pre.get_accuracy(X_test_enc_scale, Y_test, model)

# print 'SUPPORT VECTOR MACHINE (SVM)'
# model = train_svm_linear_svc(X_train[cols_of_interest], encoded_y_train)
# print 'ACCURACY before standardisation'
# print get_accuracy_rounded(X_test[cols_of_interest], encoded_y_test, model)
# print 'ACCURACY after standardisation'
# print get_accuracy(X_test_enc_scale, encoded_y_test, model)

print 'Elements before encoding'
print X_train.head()
print 'Encoding Categorical Entries'
X_train_enc, X_test_enc = pre.encoding_categorical(X_train, X_test)
print 'Elements after encoding'
print X_train_enc.head()
print 'Standardise the new encoded data'
X_train_enc_scale, X_test_enc_scale = pre.scaling(X_train_enc, X_test_enc)
print 'LOGISTIC REGRESSION after Encoding and Standardisation'
model = pre.train_logistic_regression(X_train_enc_scale, Y_train.values.ravel())
print 'ACCURACY after Encoding and Standardisation'
print pre.get_accuracy(X_test_enc_scale, Y_test, model)

print 'OneHotEncoding categorical entries'
print 'Dependents is 1 whereas distance between 0 and 3+ will be 3,' \
+ 'which is not desirable as both the distances should be similar.' \
+ 'OneHotEncoder solves this issue.'
print 'Logistic Regression before standardisation and OneHotEncoding - Accuracy:'
model = pre.train_logistic_regression(X_train, Y_train.values.ravel())
print pre.get_accuracy(X_test, Y_test, model)
print 'Logistic Regression after standardisation and after OneHotEncoding:'
X_train_hot_enc, X_test_hot_enc = pre.one_hot_encoding_categorical(X_train, X_test)
X_train_hot_enc_scale, X_test_hot_enc_scale = pre.scaling(X_train_hot_enc, X_test_hot_enc)
model = pre.train_logistic_regression(X_train_hot_enc_scale, Y_train.values.ravel())
print 'accuracy after standardisation and after OneHotEncoding:'
print pre.get_accuracy(X_test_hot_enc_scale, Y_test, model)

