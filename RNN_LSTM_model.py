#!/usr/bin/env python

import tensorflow as tf
import pandas
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

constants = {
    'alpha': 1e-3,
    'decay': 1e-5,
    'train_size': 0.75,
    'layer_sizes': {'i': 128,
                    'h1': 128,
                    'h2': 32,
                    'o': 2
                    },
    'activations': {'i': 'relu',
                    'h1': 'relu',
                    'h2': 'relu',
                    'o': 'softmax'
                    },
    'dropout': 0.3,
    'epochs': 25,
    'loss': 'sparse_categorical_crossentropy'
}


def shuffle_df(raw_data):
    return raw_data.sample(frac=1).reset_index(drop=True)


def get_df_and_labels(data):
    df = data.loc[:, data.columns != 'label']
    df = df.loc[:, data.columns != 'enum_label']
    labels = data['enum_label']
    return df, labels


def fit_labels(df):
    labels = ['male', 'female']
    df['enum_label'] = [labels.index(val) for val in df['label'].values]
    df = df.drop(columns=['label'])
    return df


def get_shuffled_divided_data(raw_data):
    data = shuffle_df(raw_data)
    data = fit_labels(data)
    df, labels = get_df_and_labels(data)

    TRAIN_SIZE = int(len(data) * constants['train_size'])

    train_df = df[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]

    test_df = df[TRAIN_SIZE:]
    test_labels = labels[TRAIN_SIZE:]

    return numpy.expand_dims(train_df, 2), train_labels, numpy.expand_dims(test_df, 2), test_labels


def main():

    # dataset
    raw_data = pandas.read_csv('dataset/voice.csv')
    x_train, y_train, x_test, y_test = get_shuffled_divided_data(raw_data)

    # model
    model = Sequential()

    model.add(LSTM(constants['layer_sizes']['i'], activation=constants['activations']['i'], input_shape=(x_train.shape[1:]), return_sequences=True))
    model.add(Dropout(constants['dropout']))

    model.add(LSTM(constants['layer_sizes']['h1'], activation=constants['activations']['h1']))
    model.add(Dropout(constants['dropout']))

    model.add(Dense(constants['layer_sizes']['h2'], activation=constants['activations']['h2']))
    model.add(Dropout(constants['dropout']))

    model.add(Dense(constants['layer_sizes']['o'], activation=constants['activations']['o']))

    # create optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=constants['alpha'], decay=constants['decay'])

    # compile model
    model.compile(optimizer=opt,
                  loss=constants['loss'],
                  metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, epochs=constants['epochs'], validation_data=(x_test, y_test))

    model.summary()


if __name__ == '__main__':
    main()
