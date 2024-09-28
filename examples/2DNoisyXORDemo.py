#!/usr/local/bin/python3
import os
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np 

this_dir, this_filename = os.path.split(__file__)
train_data = np.loadtxt(os.path.join(this_dir, "2DNoisyXORTrainingData.txt"))
X_train = train_data[:,0:-1].reshape(train_data.shape[0], 4, 4)
Y_train = train_data[:,-1]

test_data = np.loadtxt(os.path.join(this_dir, "2DNoisyXORTestData.txt"))
X_test = test_data[:,0:-1].reshape(test_data.shape[0], 4, 4)
Y_test = test_data[:,-1]

ctm = MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)

ctm.fit(X_train, Y_train, epochs=5000)

print("Accuracy:", 100*(ctm.predict(X_test) == Y_test).mean())

Xi = np.array([[[0,1,1,0],
		[1,1,0,1],
		[1,0,1,1],
		[0,0,0,1]]])

print("\nInput Image:\n")
print(Xi)
# print("\nPrediction: %d" % (ctm.predict(Xi)))
print("\nPrediction: %d" % ctm.predict(Xi)[0])
