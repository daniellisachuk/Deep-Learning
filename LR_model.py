#!/usr/bin/env python

# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas
from datetime import datetime as dt

constants = {
    'alpha': 0.00001,
    'epochs': 500000,
    'features': 20,
    'labels': 2,
    'bin_size': 1500,
    'bins': None
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

    TRAIN_SIZE = int(len(data) * 0.7)

    train_df = df[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]

    test_df = df[TRAIN_SIZE:]
    test_labels = labels[TRAIN_SIZE:]

    return train_df, train_labels, test_df, test_labels


def print_stats(acc, cost_hist, training_end_time, training_start_time):
    # ready the stat output
    first_cost = cost_hist[0]
    last_cost = cost_hist[-1]
    avg_cost = sum(cost_hist) / len(cost_hist)
    cost_hist.sort()
    lowest_cost = cost_hist[0]
    highest_cost = cost_hist[-1]
    training_time = training_end_time - training_start_time
    print('Finished Running\n')
    print('\tRunning Details:\n'
          '\t\tNumber of Epochs Set To : {}\n'.format(constants['epochs']) +
          '\t\tNumber of Bins Set To : {}\n'.format(constants['bins'] + 1) +
          '\t\tSize of Bin Per Epoch : {}\n'.format(constants['bin_size']) +
          '\t\tTotal Training Cycles : {}\n'.format(constants['epochs'] * (constants['bins'] + 1)) +
          '\t\tLearning Rate Set To : {}\n'.format(constants['alpha']) +
          '\t\tTotal Training Time : {}\n'.format(str(training_time))
          )
    print('\tCosts:\n'
          '\t\tFirst Recorded Cost : {}\n'.format(first_cost) +
          '\t\tLast Recorded Cost : {}\n'.format(last_cost) +
          '\t\tAverage Cost : {}\n'.format(avg_cost) +
          '\t\tLowest Recorded Cost : {}\n'.format(lowest_cost) +
          '\t\tHighest Recorded Cost : {}\n'.format(highest_cost)
          )
    print('\tAccuracy:\n'
          '\t\tFinal Accuracy: {} %\n'.format(acc)
          )


def main():
    raw_data = pandas.read_csv('dataset/voice.csv')

    x_o_train, y_o_train, x_o_test, y_o_test = get_shuffled_divided_data(raw_data)

    # initialize input and output vectors
    X = tf.placeholder(tf.float32, [None, constants['features']])
    Y = tf.placeholder(tf.float32, [None, constants['labels']])

    # initialize weights and biases randomly
    W = tf.Variable(0.001 * np.random.randn(constants['features'], constants['labels']).astype(np.float32))
    b = tf.Variable(0.001 * np.random.randn(constants['labels']).astype(np.float32))

    hyp = tf.nn.softmax(tf.add(tf.matmul(X, W), b))

    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(hyp, 1e-10, 1.0))))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=constants['alpha']).minimize(cost)

    init = tf.global_variables_initializer()

    # calculate bin size for training
    constants['bins'] = int(x_o_train.shape[0] / constants['bin_size'])

    training_start_time = dt.now()

    # start tensor session
    with tf.Session() as sess:
        y_o_train = sess.run(tf.one_hot(y_o_train, constants['labels']))
        y_o_test = sess.run(tf.one_hot(y_o_test, constants['labels']))

        sess.run(init)

        cost_hist = []

        # model training (optimizer)
        for epoch in range(constants['epochs']):
            for bi in range(constants['bins']):

                start_point = bi * epoch
                end_point = start_point + constants['bin_size']

                x = x_o_train[start_point: end_point]
                y = y_o_train[start_point: end_point]

                sess.run(optimizer, feed_dict={X: x, Y: y})
                c = sess.run(cost, feed_dict={X: x, Y: y})

            if (epoch % 500 is 0 and epoch is not 0) or (epoch is constants['epochs'] - 1):
                cost_hist.append(c)
                print('\rEpoch: {} Cost: {}'.format(str(epoch), str(c)))
                # print('\rW: {}, b: {}'.format(W.eval(sess), b.eval(sess)))

        training_end_time = dt.now()

        # model testing
        correct_prediction = tf.equal(tf.argmax(hyp, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = accuracy.eval({X: x_o_test, Y: y_o_test}) * 100

        print_stats(acc, cost_hist, training_end_time, training_start_time)

        print('Confusion Matrix:')

        # confusion matrix
        conf_mat = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(hyp, 1), num_classes=2)
        conf_mat_to_print = sess.run(conf_mat, feed_dict={X: x_o_test, Y: y_o_test})
        print(conf_mat_to_print)


if __name__ == '__main__':

    main()

