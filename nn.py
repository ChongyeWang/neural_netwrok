import h5py
import numpy as np
from sklearn.metrics import classification_report

"""
Load the data.
"""
def load_data(filename):
    data = h5py.File(filename, 'r')
    x_train = np.float32(data['x_train'][:])
    y_train = np.int32(np.array(data['y_train'][:, 0]))
    x_test = np.float32(data['x_test'][:])
    y_test = np.int32(np.array(data['y_test'][:, 0]))
    data.close()
    return x_train.T, y_train, x_test.T, y_test

"""
Implement the one_hot_encoding.
"""
def one_hot_encoding(y, dim):
	y = y.reshape(1, dim)
	y = np.eye(10)[y.astype('int32')]
	y = y.T.reshape(10, dim)
	return y

"""
The sigmoid function used as activation fuction.
"""
def sigmoid(z):
	return 1 / ( 1 + np.exp(-z))

"""
Calculate the cross entropy loss.
"""
def cross_entropy_loss(y, y_hat):
	return -(1.0 / y.shape[1]) * np.sum(np.dot(y, np.log(y_hat)))

"""
Forward process.
"""
def forward(x, w1, b1, w2, b2):
	Z1 = np.dot(w1, x) + b1
	A1 = sigmoid(Z1)
	Z2 = np.dot(w2, A1) + b2
	A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
	return Z1, A1, Z2, A2

"""
Backward process.
"""
def backward(x, y, Z1, A1, Z2, A2, w1, b1, w2, b2, batch_distance):
	dZ2 = A2 - y
	dW2 = (1.0 / batch_distance) * np.dot(dZ2, A1.T)
	db2 = (1.0 / batch_distance) * np.sum(dZ2, axis=1, keepdims=True)
	dA1 = np.dot(w2.T, dZ2)
	dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
	dW1 = (1.0 / batch_distance) * np.dot(dZ1, x.T)
	db1 = (1.0 / batch_distance) * np.sum(dZ1, axis=1, keepdims=True)
	return dW1, db1, dW2, db2


def main():
	x_train, y_train, x_test, y_test = load_data('MNISTdata.hdf5')
	orig_x_train, orig_y_train, orig_x_test, orig_y_test = x_train[:], y_train[:], x_test[:], y_test[:]
	y_train = one_hot_encoding(y_train, len(y_train))
	y_test = one_hot_encoding(y_test, len(y_test))
	index = np.random.permutation(60000)
	x_train, y_train = x_train[:, index], y_train[:, index]
	np.random.seed(0)
	w1 = np.random.randn(64, 784) / 28.0
	b1 = np.zeros((64, 1)) / 28.0
	w2 = np.random.randn(10, 64) / 8.0
	b2 = np.zeros((10, 1)) / 8.0
	dW1_m = np.zeros(w1.shape)
	db1_m = np.zeros(b1.shape)
	dW2_m = np.zeros(w2.shape)
	db2_m = np.zeros(b2.shape)
	epochs = 20
	batch_size = 128
	lr = 1
	for epoch in range(epochs):
		index = np.random.permutation(60000)
		shuffled_x_train, shuffled_y_train = x_train[:, index], y_train[:, index]
		batches = int(60000 / 128) + 1
		for i in range(batches):
			start = i * batch_size
			end = min(start + batch_size, x_train.shape[1] - 1)
			selected_x = shuffled_x_train[:, start:end]
			selected_y = shuffled_y_train[:, start:end]
			batch_distance = end - start
			Z1, A1, Z2, A2 = forward(selected_x, w1, b1, w2, b2)
			dW1, db1, dW2, db2 = backward(selected_x, selected_y, Z1, A1, Z2, A2, w1, b1, w2, b2, batch_distance)
			momentum_ratio = 0.9
			dW1_m = (momentum_ratio * dW1_m + (1.0 - momentum_ratio) * dW1)
			db1_m = (momentum_ratio * db1_m + (1.0 - momentum_ratio) * db1)
			dW2_m = (momentum_ratio * dW2_m + (1.0 - momentum_ratio) * dW2)
			db2_m = (momentum_ratio * db2_m + (1.0 - momentum_ratio) * db2)
			w1 = w1 - lr * dW1_m
			b1 = b1 - lr * db1_m
			w2 = w2 - lr * dW2_m
			b2 = b2 - lr * db2_m

	#Test 
	Z1, A1, Z2, A2 = forward(x_test, w1, b1, w2, b2)
	predictions = np.argmax(A2, axis=0)
	total_correct = 0
	for i in range(len(orig_y_test)):
		total_correct += predictions[i] == orig_y_test[i]
	test_accuracy = float(total_correct) / len(orig_y_test)
	print(test_accuracy) #0.9763

if __name__ == "__main__":
	main()
