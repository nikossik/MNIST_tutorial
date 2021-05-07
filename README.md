# MNIST tutorial

**MNIST is a dataset containing tiny gray-scale images, each showing a handwritten digit, that is, 0, 1, 2, …, 9. Your mission is to analyze such an image, and tell what digit is written there. The dataset looks like this:**

![MNIST](https://miro.medium.com/max/1600/0*9jCey4wywZ4Os7hF.png)

#### Handwritten digit recognition, in general, is a realistic task. The MNIST dataset is also not particularly small: it contains 60,000 images in the training set and 10,000 in the test set. Each image has a resolution of 28x28, totaling 28²=784 features — a rather high dimensionality. So why is MNIST a “Hello World” example? One reason is that it is surprisingly easy to obtain decent accuracy, like 90%, even with a weak or poorly designed machine learning model. A practical setting, seemingly challenging task, high accuracy with little work — a perfect combination for beginners.

---

# Code

### Imports
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
```

### Then we import MNIST data and get its shape
```python
nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
```

### Format data for training 
```python
X_train = X_train.reshape(-1, 784).astype(np.float32)
X_test = X_test.reshape(-1, 784).astype(np.float32)

X_train /= 255.0
X_test /= 255.0
```
```python
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
```
### Build the neural network
* In the next cell you are supposed to implement your architecture of **Feed Forward Network**
* Use the following Keras layers:
    * `Input` - layer for the input node, [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Input)
    * `Dense` - fully connected layer (activation can be included as param), [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
    * `Activation` - activation layer, [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation)
    * `Dropout` (optional) - dropout regularizer, [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
    
    
* Make sure your last layer has `nb_classes` neurons and `softmax` activation which allows you to model probabilistic distribution over all classes 
* Softmax activation:
![softmax](https://i.ytimg.com/vi/o6HrH2EMD-w/maxresdefault.jpg)

```python
input_shape = X_train.shape[1:] # batch dim is not included

input_layer = Input(shape=input_shape)
fc0 = Dense(64, activation='elu')(input_layer)
dropout0 = Dropout(0.9)(fc0)
fc1 = Dense(128, activation='relu')(dropout0)
dropout1 = Dropout(0.7)(fc1)
fc2 = Dense(32, activation='elu')(dropout1)
output_layer = Dense(10, activation='softmax')(fc2)

model = Model(inputs=input_layer, outputs=output_layer)
```

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Let's see model summary
```python
model.summary()
```
```
Output:
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 784)]             0         
_________________________________________________________________
dense (Dense)                (None, 64)                50240     
_________________________________________________________________
dropout (Dropout)            (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               8320      
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                4128      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330       
=================================================================
Total params: 63,018
Trainable params: 63,018
Non-trainable params: 0
_________________________________________________________________
```
```python
# get history from training info
hist_loss = training_info.history['loss']
hist_val_loss = training_info.history['val_loss']
hist_acc = training_info.history['accuracy']
hist_val_acc = training_info.history['val_accuracy']

# plot losses
plt.figure(figsize=(8, 6))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist_loss, color='red', label='train_loss')
plt.plot(hist_val_loss, color='green', label='val_loss')
plt.xlim(0, len(hist_loss) - 1)
plt.legend(loc='best')
plt.grid(True)
plt.show()

# plot metrics
plt.figure(figsize=(8, 6))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(hist_acc, color='orange', label='train_acc')
plt.plot(hist_val_acc, color='purple', label='val_acc')
plt.xlim(0, len(hist_acc) - 1)
plt.legend(loc='best')
plt.grid(True)
plt.show()
```
### Finally, evaluate its performance
```python
train_acc = model.evaluate(X_train, y_train)[1]
test_acc = model.evaluate(X_test, y_test)[1]

print('Train accuracy:', train_acc)
print('Test accuracy:', test_acc)
```
```
Output:
60000/60000 [==============================] - 1s 16us/sample - loss: 0.4260 - accuracy: 0.8814
10000/10000 [==============================] - 0s 17us/sample - loss: 0.4247 - accuracy: 0.8787
Train accuracy: 0.8814167
Test accuracy: 0.8787
```
### Inspecting the output
* It's always a good idea to inspect the output and make sure everything looks sane. Here we'll look at some examples it gets right, and some examples it gets wrong.

```python
predicted_classes = np.argmax(model.predict(X_test), axis=1)

correct_indices = np.nonzero(predicted_classes == np.argmax(y_test, axis=1))[0]
incorrect_indices = np.nonzero(predicted_classes != np.argmax(y_test, axis=1))[0]
```
```python
print('Sample correct predictions')
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], np.argmax(y_test[correct])))
    
plt.show()

print('Sample wrong predictions')
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], np.argmax(y_test[incorrect])))
    
plt.show()
```
### That's all

