#  https://raw.githubusercontent.com/ianmcloughlin/jupyter-teaching-notebooks/master/mnist.ipynb

import gzip
import os.path
import tkinter as tk
from random import randint
from tkinter import filedialog

import keras as kr
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as pre
from keras.preprocessing import image

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
   #print(i, encoder.transform([i]))
# prints out the arrays

if os.path.isfile('data/model.h5'): 
        model = kr.models.load_model('data/model.h5')
# if model already exist uses it
else:
    model.fit(inputs, outputs, epochs=15, batch_size=100)
    model.save("data/model.h5")
    #makes model and saves it 

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8)
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

outcome = (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()
print("\nModel  is", outcome/100,"% Accurate\n")
print("\nModel has been created or loaded into memory")


#model.predict(test_img[5:6])
#plt.imshow(test_img[5].reshape(28, 28), cmap='gray'

def newmethod791():
    amm = int(input(" How many tests would you like to run ? "))
    from random import randint
    for i in range(amm):
        print("Test Number : ", i+1,"\n")
        x = randint(0, 9999)

        print("The random index: ", x, "\n")
        print("The result array: ")
        test = model.predict(test_img[x:x+1])
        # Print the result array
        print(test, "\n")
        # Get the maximum value from the machine predictions
        pred_result = test.argmax(axis=1)

        print("program predicted : =>> ",  pred_result)
        print(" number is : =>> ", test_lbl[x:x+1])
        print("===================")

def loadImage():
    root = tk.Tk()
    root.withdraw()
    #https://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python
    file_path = filedialog.askopenfilename()# opens file select window
    img = image.load_img(path=file_path,color_mode = "grayscale",target_size=(28,28,1))
    image1 = np.array(list(image.img_to_array(img))).reshape(1, 784).astype(np.uint8) / 255.0
    # shapes array 
    plt.imshow(img)
    plt.show()
    # plots and displays image
    test = model.predict(image1)
    # runs test of image on model
    print("program has predicted : ", test.argmax(axis=1))
#https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b

print("Load an image on your system")

opt=True
while opt:
    print("============================")
    print("""        1 to load image
        2 to run test
        3 to exit """)
    opt= input(" What would you like to do ? ")
    print("============================")
    #https://stackoverflow.com/questions/19964603/creating-a-menu-in-python

    if opt == "1":
        loadImage()
    elif opt == "2":
        newmethod791()
    elif opt == "3":
        exit()
