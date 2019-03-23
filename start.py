import preprocessing as pre
import pandas as pd
from const import CONST
from three_l_nn_tf import ThreeLayerNN as Tlnntf
from conv_nn import ConvNN


X_test = pd.read_csv(CONST.X_TEST)
Y_train = pd.read_csv(CONST.Y_TRAIN)
Y_test = pd.read_csv(CONST.Y_TEST)
X_train = pd.read_csv(CONST.X_TRAIN)
X_train_minmax, X_test_minmax = pre.scaling(X_train[CONST.COLS], X_test[CONST.COLS])
encoded_y_train, encoded_y_test = pre.encoding_categorical(Y_train, Y_test)
X_train_enc_scale, X_test_enc_scale = pre.scaling(X_train[CONST.COLS], X_test[CONST.COLS])
X_train_enc, X_test_enc = pre.encoding_categorical(X_train, X_test)
X_train_enc_scale, X_test_enc_scale = pre.scaling(X_train_enc, X_test_enc)
X_train_hot_enc, X_test_hot_enc = pre.one_hot_encoding_categorical(X_train, X_test)
X_train_hot_enc_scale, X_test_hot_enc_scale = pre.scaling(X_train_hot_enc, X_test_hot_enc)


def three_l_nn_tf(l1, l2, l3, mi, step, stddev, epochs, batch_size):
    threelnntf = Tlnntf(l1, l2, l3)
    w, b = threelnntf.init_rand_norm_weights_biases(stddev, threelnntf.nn_structure)
    threelnntf.train_nn(X_train_enc_scale[:200], encoded_y_train.values.ravel(), w, b, mi, step, epochs, batch_size, stddev)
        

tot_nodes_l1 = 43 # 43 features for 1 packets; 43=6*7+1
tot_nodes_l2 = 20
tot_nodes_l3 = 10
max_iteration = 3000
stddev = 0.03
step = 0.25
epochs = 10
batch_size = 50

# accuracies, costs = three_l_nn_tf(tot_nodes_l1, tot_nodes_l2, 
# tot_nodes_l3, max_iteration, step, stddev, epochs, batch_size)
# print "ACCURACIES: "
# print accuracies
# print "\nCOSTS: "
# print costs
# print "\n"


input_chans = 1 # for simplicity (it should be RGB instead (and therefore = 3) later on)
num_filters = [input_chans, 32, 64]
convnn = ConvNN(tot_nodes_l1, tot_nodes_l3)
filter_width = 3
filter_height = 3
pool_width = 2
pool_height = 2
# after two rounds of downsampling starting with 6*8
# there will be 3*4 samples first and 1*2 samples at the end
final_grid_width = 1
final_grid_height = 2
train_x = X_train_enc_scale[:200]
train_y = encoded_y_train.values.ravel()
test_x = X_test_enc_scale[:200]
test_y = encoded_y_test.values.ravel()
img_width = 6
img_height = 8
learning_rate = 0.0001
# convnn.train_nn(num_filters, filter_width, filter_height, 
#                 pool_width, pool_height, final_grid_width, 
#                 final_grid_height, test_x, test_y, train_x, 
#                 train_y, learning_rate, epochs, batch_size, 
#                 img_width, img_height)
print 'train_x[:5]'
print train_x[:20]
print 'train_y[:5]'
print train_y[:1000]
print 'test_x[:5]'
print test_x[:20]
print 'test_y tot'
#print test_y[9000:10000]
count6 = 0
count0 = 0
for y in test_y:
    if y != 6:
        count6 += 1
    if y == 0:
        count0 += 1
print len(test_y)
print 'test_y non 6 (generic) tot'
print count6
print 'test_y = 6 tot'
print len(test_y) - count6
print 'test_y = 0 (none)'
print count0