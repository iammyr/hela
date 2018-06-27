import pandas as pd

# Trains a KNN model with X data and Y corresponding categories.
# Tests the accuracy of the trained model using X_test and Y_test.
# It only considers data under the cols-titled columns (cols is an array of column titles
# to consider).
# returns the traind KNN model.
# Note: y_train needs to be a 1-dimensional vector.
def train_knn(x_train, y_train, cols):
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=len(cols))
    knn.fit(x_train, y_train)
    return knn


def get_accuracy(x_test, y_test, model):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, model.predict(x_test))


def get_accuracy_rounded(x_test, y_test, model):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, model.predict(x_test).round())


# Scaling down both train and test data set (using MinMaxScaler, get values btw 0 and 1)
def scaling(x_train, x_test):
    from sklearn.preprocessing import MinMaxScaler
    min_max = MinMaxScaler()
    return min_max.fit_transform(x_train), min_max.fit_transform(x_test)


# STANDARDISING, i.e., Standardization (or Z-score normalization)
# is the process where the features are rescaled so that they will
# have the properties of a standard normal distribution with mu=0 and sigma=1,
# where mu is the mean (average) and sigma is the standard deviation from the mean
def standardising(x_train, x_test):
    from sklearn.preprocessing import scale
    return scale(x_train), scale(x_test)


def train_linear_regression(x_train, y_train):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr


# Note: y_train needs to be a 1-dimensional vector.
def train_logistic_regression(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(penalty='l2', C=.01)
    log.fit(x_train, y_train)
    return log


def train_svm_linear_svc(x_train, y_train):
    from sklearn.svm import SVC
    svc = SVC(kernel='linear', C=.01)
    svc.fit(x_train, y_train)
    return svc


def encoding_categorical(x_train, x_test):
    from sklearn.preprocessing import LabelEncoder
    X_train_enc = x_train
    X_test_enc = x_test
    le = LabelEncoder()
    for col in x_test.columns.values:
        # Encoding only categorical variables
        if x_test[col].dtypes == 'object':
            # Use the whole data to form an exhaustive list of levels
            data = x_train[col].append(x_test[col])
            le.fit(data.values)
            X_train_enc[col] = le.transform(x_train[col])
            X_test_enc[col] = le.transform(x_test[col])
    return X_train_enc, X_test_enc


def one_hot_encoding_categorical(x_train, x_test):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    X_train_hot_enc = x_train
    X_test_hot_enc = x_test

    for col in x_test.columns.values:
        # Encoding only categorical variables
        if x_test[col].dtypes=='object':    
            # creating an exhaustive list of all possible categorical values
            data = x_train[[col]].append(x_test[[col]])
            enc.fit(data)
            # Fitting One Hot Encoding on train data
            temp = enc.transform(x_train[[col]])
            # Changing the encoded features into a data frame with new column names
            temp = pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
                    .value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the X_train data frame
            temp = temp.set_index(x_train.index.values)
            # adding the new One Hot Encoded variables to the train data frame
            X_train_hot_enc = pd.concat([X_train_hot_enc, temp],axis=1)
            # fitting One Hot Encoding on test data
            temp = enc.transform(x_test[[col]])
            # changing it into data frame and adding column names
            temp = pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
                    .value_counts().index])
            # Setting the index for proper concatenation
            temp = temp.set_index(x_test.index.values)
            # adding the new One Hot Encoded variables to test data frame
            X_test_hot_enc = pd.concat([X_test_hot_enc, temp],axis=1)
    return X_train_hot_enc, X_test_hot_enc
