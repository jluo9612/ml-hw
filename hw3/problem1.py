import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Confusion matrix plotting module
from plot import plot_confusion_matrix

# Load data set
(x_traino, y_train),(x_testo, y_test) = mnist.load_data()
x_train = np.reshape(x_traino,(60000,28*28))
x_test = np.reshape(x_testo,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0


logreg = LogisticRegression(solver='saga', multi_class='multinomial',max_iter = 100,verbose=2)
est = logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

# Accuracy
score = accuracy_score(y_test, y_pred)
print("Accuracy score: %.4f" % score)

# Show confusion matrix for prediction
conf = plot_confusion_matrix(y_test, y_pred, np.array(range(10)), normalize=True)
plt.show()