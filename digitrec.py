#  https://raw.githubusercontent.com/ianmcloughlin/jupyter-teaching-notebooks/master/mnist.ipynb

import keras as kr
import gzip
import numpy as np
import sklearn.preprocessing as pre
# imports needed

model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))
model.add(kr.layers.Dense(units=1000, activation='relu'))
model.add(kr.layers.Dense(units=1000, activation='relu'))
model.add(kr.layers.Dense(units=1000, activation='relu'))

# Add a 10 neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# Unzips the files and reads in as bytes
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()
# reads them into memory 
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

inputs = train_img.reshape(60000, 784)/255
# For encoding categorical variables.
encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

print(train_lbl[0], outputs[0])

#for i in range(10):
   # print(i, encoder.transform([i]))

model.fit(inputs, outputs, epochs=15, batch_size=100)

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8)
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

(encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()

#model.predict(test_img[5:6])
#plt.imshow(test_img[5].reshape(28, 28), cmap='gray'
# 
print("==========================")

from random import randint
for i in range(10):
    print("Test Case No:", i+1)
    print("==========================")
    x = randint(0, 9999)
    print("Index: ", x)
    print("Result array: ")
    test = model.predict(test_img[x:x+1])
    # Prints the array
    print(test)
    print("The number was : ", test_lbl[x:x+1])
    # Get the maximum value from the machine predictions
    pred_result = test.argmax(axis=1)

    print("Program prediction is : ",  pred_result)
    print("==========================")