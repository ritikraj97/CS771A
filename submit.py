import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def binary_to_decimal(data, row):
    binary1 = ''
    binary2 = ''

    for i in data[row][64:68]:
        binary1 += str(int(i))

    for j in data[row][68:72]:
        binary2 += str(int(j))

    p = int(binary1, 2)
    q = int(binary2, 2)

    return [p, q]

def create_feature(data):
    for challenge in data:
        np.append(challenge, 1.0)
    return data

def transform_train_data(data):
	n = np.shape(data)[0]
	trans_data = []
	for row in range(n):
		challenge = np.append(np.append(data[row][0:64], binary_to_decimal(data, row)), data[row][-1])
		trans_data.append(challenge)
	trans_data = np.array(trans_data)
	return trans_data


def transform_test_data(data):
	n = np.shape(data)[0]
	trans_data = []
	for row in range(n):
		challenge = np.append(data[row][0:64], binary_to_decimal(data, row))
		trans_data.append(challenge)
	trans_data = np.array(trans_data)
	return trans_data

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	model = {}
	Z_train = transform_train_data(Z_train)

	train_data = {}

	for challenge in Z_train:
		p = int(challenge[64])
		q = int(challenge[65])
		key = None
		if p < q:
			key = str(p) + '$' + str(q)
		else:
			key = str(q) + '$' + str(p)

		if train_data.get(key) is None:
			train_data[key] = np.empty((0, 65), float)

		challenge = np.delete(challenge, [64, 65])
		if p < q:
			train_data[key] = np.append(train_data[key], np.array([challenge]), axis=0)
		else:
			challenge[-1] = 1.0 - challenge[-1]
			train_data[key] = np.append(train_data[key], np.array([challenge]), axis=0)

	for key, data in train_data.items():
		if model.get(key) is None:
			model[key] = LogisticRegression(C=21, max_iter=500, tol=1e-2)
		X = create_feature(data[:, :-1])
		y = data[:, -1]

		model[key].fit(X, y)

	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	X_tst = transform_test_data(X_tst)
	predicted_response = []

	for challenge in X_tst:
		p = int(challenge[64])
		q = int(challenge[65])

		challenge = np.delete(challenge, [64, 65])

		if (p < q):
			key = str(p) + '$' + str(q)
			predicted_response.append((model[key].predict(create_feature([challenge])))[0])
		else:
			key = str(q) + '$' + str(p)
			predicted_response.append(1.0 - (model[key].predict(create_feature([challenge])))[0])

	predicted_response = np.array(predicted_response)
	return predicted_response
