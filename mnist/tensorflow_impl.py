"""
Tensorflow implementation of a Convolutional Network
for the MNIST dataset. Adapted based on the tutorial:

    https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html#deep-mnist-for-experts

"""

import tensorflow as tf
import time

from data import Data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def forward(image, keep_prob):
    # Reshape to 4D
    image_4d = tf.reshape(image, [-1,28,28,1])

    # Conv1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(image_4d, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Conv2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # FC1 (with dropout)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # FC2
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def build_graph():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = forward(x, keep_prob)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    return x, y_, keep_prob, train_step, num_correct

def main():
    NUM_EPOCHS = 30
    data = Data(batch_size=64, validation_size=6000)

    session = tf.Session()
    input_placeholder, output_placeholder, keep_prob_placeholder, train_step_f, num_correct_f = build_graph()
    noop = tf.no_op()

    session.run(tf.initialize_all_variables())

    def accuracy(data_iter, train=False):
        num_total   = 0
        num_correct = 0
        for batch_x, batch_y in data_iter:
            print 'batch'
            batch_num_correct, _ = session.run(
                [num_correct_f, train_step_f if train else noop],
                {
                    input_placeholder:     batch_x,
                    output_placeholder:    batch_y,
                    keep_prob_placeholder: 0.5 if train else 1.0,
                })
            num_correct += batch_num_correct
            num_total   += len(batch_x)

        return float(num_correct) / float(num_total)

    train_total_time_sum = 0
    for epoch in range(NUM_EPOCHS):
        train_start_time = time.time()
        train_accuracy    = accuracy(data.iterate_train(),    train=True)
        train_total_time = time.time() - train_start_time
        train_total_time_sum += train_total_time

        validate_accuracy = accuracy(data.iterate_validate(), train=False)

        print ("Training epoch number %d:" % (epoch,))
        print ("    Time to train           = %.3f s" % (train_total_time))
        print ("    Training set accuracy   = %.1f %%" % (100.0 * train_accuracy,))
        print ("    Validation set accuracy = %.1f %%" % (100.0 * validate_accuracy,))
        print ("")
    print ("Training done.")

    test_accuracy = accuracy(data.iterate_test(), train=False)
    print ("    Average time per training epoch = %.3f s" % (train_total_time_sum / NUM_EPOCHS,))
    print ("    Test set accuracy               = %.1f %%" % (100.0 * test_accuracy,))

if __name__ == '__main__':
    main()
