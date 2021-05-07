# MNIST tutorial

**MNIST is a dataset containing tiny gray-scale images, each showing a handwritten digit, that is, 0, 1, 2, …, 9. Your mission is to analyze such an image, and tell what digit is written there. The dataset looks like this:**

![MNIST](https://miro.medium.com/max/1600/0*9jCey4wywZ4Os7hF.png)

### Handwritten digit recognition, in general, is a realistic task. The MNIST dataset is also not particularly small: it contains 60,000 images in the training set and 10,000 in the test set. Each image has a resolution of 28x28, totaling 28²=784 features — a rather high dimensionality. So why is MNIST a “Hello World” example? One reason is that it is surprisingly easy to obtain decent accuracy, like 90%, even with a weak or poorly designed machine learning model. A practical setting, seemingly challenging task, high accuracy with little work — a perfect combination for beginners.

# Code

### Imports
***
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
***

### Then we import MNIST data and get its shape
'''
nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
'''

### Format data for training 
'''
X_train = X_train.reshape(-1, 784).astype(np.float32)
X_test = X_test.reshape(-1, 784).astype(np.float32)

X_train /= 255.0
X_test /= 255.0
'''
'''
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
'''
