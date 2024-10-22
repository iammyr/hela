import preprocessing as pre
import pandas as pd
from const import CONST
from three_l_nn import ThreeLayerNN
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
from three_l_nn_tf import ThreeLayerNN as Tlnntf


X_test = pd.read_csv(CONST.X_TEST)
Y_train = pd.read_csv(CONST.Y_TRAIN)
Y_test = pd.read_csv(CONST.Y_TEST)
X_train = pd.read_csv(CONST.X_TRAIN)

# feature scaling, i.e., limit the range of variables 
# so that they can be compared on common grounds
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
print encoded_y_test[:50].values.ravel()
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


def three_l_nn(l1, l2, l3, mi, s):
    # 43=n 43 features 43 input signals
    # 9=output; Attack categories
    # 1 Fuzzers, 
    # 2 Analysis, 
    # 3 Backdoors, 
    # 4 DoS 
    # 5 Exploits, 
    # 6 Generic, 
    # 7 Reconnaissance, 
    # 8 Shellcode
    # 9 Worms
    # 0 None
    # Hidden Layer in 3-layers NN has 20 nodes
    # nn_structure = [43, 20, 10]
    threelnn = ThreeLayerNN(l1, l2, l3)
    # 1) randomly initialise weights and biases for all layers
    weights, biases = threelnn.init_weights_biases()
    weights, biases, avg_cost_curve = threelnn.train_nn(X_train_hot_enc_scale[:50], encoded_y_train.values.ravel(), weights, biases, mi, s)
    predicted_classes = threelnn.test_nn(weights, biases, X_test_hot_enc_scale[:50])
    acc_score = accuracy_score(Y_test[:50].values.ravel(), predicted_classes) * 100
    return avg_cost_curve, acc_score, predicted_classes

def display(avg_cost_curve, acc_score, predicted_classes):
    plt.plot(avg_cost_curve)
    plt.ylabel('Average Cost Function')
    plt.xlabel('Iteration number')
    plt.show()
    print 'expected:'
    print Y_test[:50].values.ravel()
    print 'actual:'
    print predicted_classes
    print 'ACCURACY SCORE for custom 3-layers Neural Network'
    print acc_score


def three_l_nn_tf(l1, l2, l3, mi, step, stddev, epochs, batch_size):
    threelnntf = Tlnntf(l1, l2, l3)
    w, b = threelnntf.init_rand_norm_weights_biases(stddev, threelnntf.nn_structure)
    threelnntf.train_nn(X_train_enc_scale[:200], encoded_y_train.values.ravel(), w, b, mi, step, epochs, batch_size, stddev)
        

tot_nodes_l1 = 43
tot_nodes_l2 = 20
tot_nodes_l3 = 10
max_iteration = 3000
stddev = 0.03
step = 0.25
epochs = 10
batch_size = 100
avg_cost_curve, acc_score, predicted_classes = three_l_nn(tot_nodes_l1, tot_nodes_l2, tot_nodes_l3, max_iteration, step)
display(avg_cost_curve, acc_score, predicted_classes)

accuracies, costs = three_l_nn_tf(tot_nodes_l1, tot_nodes_l2, tot_nodes_l3, max_iteration, step, stddev, epochs, batch_size)
print "ACCURACIES: "
print accuracies
print "\nCOSTS: "
print costs
print "\n"



