def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f(self)
    return property(fget, fset)


class _Const(object):
    @constant
    def X_TEST(self):
        return "~/hela_wspace/encrypted-network-datasets/sets/x_test.csv"

    @constant
    def X_TRAIN(self):
        return "~/hela_wspace/encrypted-network-datasets/sets/x_train.csv"

    @constant
    def Y_TRAIN(self):
        return "~/hela_wspace/encrypted-network-datasets/sets/y_train.csv"

    @constant
    def Y_TEST(self):
        return "~/hela_wspace/encrypted-network-datasets/sets/y_test.csv"

    @constant
    def COLS(self):
        return ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload']


CONST = _Const()