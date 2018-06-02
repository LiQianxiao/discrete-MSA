def read_cifar10_data():
    '''
    require Theano==0.80 version and pylearn2
    '''
    from pylearn2.datasets.zca_dataset import ZCA_Dataset
    from pylearn2.utils import serial
    train_set_size = 45000
    preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
    train_set = ZCA_Dataset(
        preprocessed_dataset=serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"),
        preprocessor = preprocessor,
        start=0, stop = train_set_size)
    valid_set = ZCA_Dataset(
        preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"),
        preprocessor = preprocessor,
        start=45000, stop = 50000)
    test_set = ZCA_Dataset(
        preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/test.pkl"),
        preprocessor = preprocessor)

    import pdb; pdb.set_trace()
    train_set.X = train_set.X.reshape(-1,3,32,32)
    valid_set.X = valid_set.X.reshape(-1,3,32,32)
    test_set.X = test_set.X.reshape(-1,3,32,32)

    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)

    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])

    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    train_set.X = train_set.X.astype(np.float32)
    valid_set.X = valid_set.X.astype(np.float32)
    test_set.X = test_set.X.astype(np.float32)

    train_set.y = train_set.y.astype(np.float32)
    valid_set.y = valid_set.y.astype(np.float32)
    test_set.y = test_set.y.astype(np.float32)

    x_train = train_set.X
    y_train = train_set.y

    x_validate = valid_set.X
    y_validate = valid_set.y

    x_test = test_set.X
    y_test = test_set.y

    # Reorder the indices of the array.
    x_train = x_train.transpose([0, 2, 3, 1])
    x_validate = x_validate.transpose([0, 2, 3, 1])
    x_test = x_test.transpose([0, 2, 3, 1])
