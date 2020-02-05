import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
mnist = keras.datasets.mnist

# Confusion matrix plotting module
from plot import plot_confusion_matrix

# Load data set
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train,(60000, 28, 28))
x_test = np.reshape(x_test,(10000, 28, 28))

# Normalize pixel values (0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Set up layers
nn = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

# Compile model
nn.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

# Fit model
nn.fit(x_train, y_train, epochs=5)

# Evaluate test accuracy
test_loss, test_acc = nn.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:%.4f' % test_acc)

# Predict 
pred = nn.predict(x_test)

# Take max confidence
pred_max = []
for p in pred:
	pred_max.append(np.argmax(p))

# Show confusion matrix
conf = plot_confusion_matrix(y_test, pred_max, np.array(range(10)), normalize=True)
plt.show()
