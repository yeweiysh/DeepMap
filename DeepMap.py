import numpy as np
import time, math
import scipy.io as sci
import os
import graph_canonicalization as gc
from scipy.sparse import load_npz

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Input, Flatten, Dense, Dropout, Conv1D,\
    Lambda, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Activation, Concatenate
from keras import regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import multi_gpu_model
from multiprocessing import Pool
from keras.utils import Sequence

# config = ConfigProto()
# sess = tf.Session(config=config)

#config = tf.ConfigProto(device_count={'CPU': 20})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def get_model(filter_size, num_instance, feature_size, num_class):
    x = Input((num_instance * filter_size, feature_size))
    flow = Conv1D(filters=32, kernel_size=filter_size, strides=filter_size, padding='same', use_bias=True)(x)
    flow = Activation('relu')(flow)

    flow = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', use_bias=True)(flow)
    flow = Activation('relu')(flow)

    flow = Conv1D(filters=8, kernel_size=1, strides=1, padding='same', use_bias=True)(flow)
    flow = Activation('relu')(flow)

    flow = Lambda(lambda x: K.sum(x, axis=1))(flow)
    flow = Dense(128, activation='relu', use_bias=True)(flow)
    flow = Dropout(0.5)(flow)

    predictions = Dense(num_class, activation='softmax')(flow)

    model = Model(inputs=x, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_callbacks(patience_lr):
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=patience_lr, verbose=1, min_lr=0,
                                       mode='min')

    return [reduce_lr_loss]


def train_model(X, y, train_idx, test_idx, num_sample, feature_size, num_class, batch_size, EPOCHS, filter_size):

    X_val = [X[id].toarray() for id in test_idx]
    y_val = y[test_idx, :]

    callbacks = get_callbacks(patience_lr=5)

    model = get_model(filter_size, num_sample, feature_size, num_class)
    model.summary()

    result = model.fit(x=np.array([X[id].toarray() for id in train_idx]), y=y[train_idx, :], batch_size=batch_size,
                        epochs=EPOCHS, verbose=1, callbacks=callbacks,
                        validation_data=[np.array(X_val), y_val], shuffle=True)

    val_acc = result.history['val_acc']
    acc = result.history['acc']

    return val_acc, acc


if __name__ == "__main__":
    # location to save the results
    OUTPUT_DIR = "/data/home/weiye/Wei/GCNN/results/"
    # location of the datasets
    DATA_DIR = "/data/home/weiye/Wei/GCNN/datasets/"

    # hyperparameters
    # ds_name = sys.argv[4] # dataset name
    dataset = ['Synthie', 'BZR_MD', 'COX2_MD', 'DHFR', 'PTC_MM', 'PTC_MR', 'PTC_FM', 'PTC_FR', 'ENZYMES', 'KKI', 'IMDB-BINARY', 'IMDB-MULTI']
    hasnodelabel = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    filter_size = 3
    kfolds = 10
    batch_size = 32
    EPOCHS = 200

    graphlet_size = 5
    max_h = 2

    feature_type = 1  # 1 (graphlet), 2 (SP), 3 (WL)

    for i in range(12):

        ds_name = dataset[i]  # dataset name
        filename = DATA_DIR + ds_name + '.mat'
        data = sci.loadmat(filename)
        graph_data = data['graph']
        graph_labels = data['label'].T[0]
        num_graphs = len(graph_data[0])

        num_class = len(np.unique(graph_labels))

        print("Dataset: %s\n" % (ds_name))

        hasnl = hasnodelabel[i]
        hasatt = hasnodeattribute[i]
        val_acc = np.zeros((kfolds, EPOCHS))
        acc = np.zeros((kfolds, EPOCHS))

        start = time.time()

        X, feature_size, num_sample = gc.canonicalization(ds_name, graph_data, hasnl, filter_size, feature_type, graphlet_size, max_h)
        folds = list(StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=7).split(np.zeros(num_graphs), graph_labels))

        encoder = LabelEncoder()
        encoder.fit(graph_labels)
        encoded_Y = encoder.transform(graph_labels)
        y = np_utils.to_categorical(encoded_Y)

        for j, (train_idx, test_idx) in enumerate(folds):
            print('\nFold ', j)

            scores_val_acc, scores_acc = train_model(X, y, train_idx, test_idx, num_sample, feature_size, num_class, batch_size, EPOCHS, filter_size)
            #print(scores)
            val_acc[j, :] = scores_val_acc
            acc[j, :] = scores_acc

        val_acc_mean = np.mean(val_acc, axis=0) * 100
        val_acc_std = np.std(val_acc, axis=0) * 100
        best_epoch = np.argmax(val_acc_mean)
        print("Average Accuracy: ")
        print("%.2f%% (+/- %.2f%%)" % (val_acc_mean[best_epoch], val_acc_std[best_epoch]))
        mean_acc = np.mean(acc, axis=0)
        for i in range(len(mean_acc)):
            print(mean_acc[i])
        end = time.time()

        print("eclipsed time: %g" % (end - start))
