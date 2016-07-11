"""
Dali implementation of a Convolutional Network
for the MNIST dataset.

"""

import dali as D
import time

from data import Data


def weight_variable(shape):
    return D.Tensor.gaussian(std=0.1, shape=shape)

def bias_variable(shape):
    return D.Tensor.zeros(shape=shape) + 0.1


def conv2d(x, W):
    return D.tensor.op.spatial.conv2d(
            x, W, strides=[1, 1], padding='SAME')

def max_pool_2x2(x):
    return D.tensor.op.spatial.max_pool(
        x, window=[2,2], strides=[2, 2], padding='SAME')

class Network(object):
    def __init__(self):
        # Conv1
        self.W_conv1 = weight_variable([32, 1, 5, 5])
        self.b_conv1 = bias_variable([32])

        # Conv2
        self.W_conv2 = weight_variable([64, 32, 5, 5])
        self.b_conv2 = bias_variable([64])

        # FC1 (with dropout)
        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        # FC2
        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

    def __call__(self, image, keep_prob):
        # Reshape to 4D
        image_4d = image.reshape([-1, 1, 28, 28])

        # Conv1
        h_conv1 = D.tensor.op.spatial.conv2d_add_bias(
            conv2d(image_4d, self.W_conv1).relu(),
            self.b_conv1
        )
        h_pool1 = max_pool_2x2(h_conv1)

        # Conv2
        h_conv2 = D.tensor.op.spatial.conv2d_add_bias(
            conv2d(h_pool1, self.W_conv2).relu(),
            self.b_conv2
        )
        h_pool2 = max_pool_2x2(h_conv2)

        # FC1 (with dropout)
        h_pool2_flat = h_pool2.reshape([-1, 7 * 7 * 64])
        h_fc1 = D.tensor.op.dot.dot(h_pool2_flat, self.W_fc1) + self.b_fc1[None,:]

        h_fc1_drop = D.tensor.op.dropout.dropout(h_fc1, 1.0 - keep_prob)

        # FC2
        y_conv = D.tensor.op.cost.softmax(
            D.tensor.op.dot.dot(h_fc1_drop, self.W_fc2) + self.b_fc2[None,:]
        )

        return y_conv

    def variables(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1,   self.b_fc1,
                self.W_fc2,   self.b_fc2]


def get_metrics(network, batch_x, batch_y, keep_prob):
    y_conv = network(D.Tensor(batch_x), keep_prob)

    cross_entropy = D.tensor.op.cost.cross_entropy(y_conv, D.Tensor(batch_y))

    with D.NoBackprop():
        predictions = y_conv.argmax(axis=1)
        correct     = D.Tensor(batch_y).argmax(axis=1)
        num_correct = D.array.op.binary.equals(
                predictions.w, correct.w).eval().sum().eval()

    return cross_entropy, float(num_correct) / len(batch_x)

def main():
    NUM_EPOCHS = 30
    data = Data(batch_size=64, validation_size=6000)

    network = Network()

    solver = D.tensor.solver.Adam(network.variables(), step_size=1e-4)

    def accuracy(data_iter, train=False):
        num_total   = 0
        num_correct = 0
        for batch_x, batch_y in data_iter:
            keep_prob = 0.5 if train else 1.0
            with D.NoBackprop(not train):
                cross_entropy, batch_num_correct = get_metrics(
                        network, batch_x, batch_y, keep_prob)
            if train:
                cross_entropy.grad()
                D.backward()
                solver.step(network.variables())

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
