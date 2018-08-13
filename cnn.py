# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
'''
 first argument is number of filters
 The second is the shape of filter
 
 the input_shape is decided by us irrespective of image size. we will convert the images
 into our format by force in image preprocessing
 Note- The order of input_shape is important. In tensorflow it is as above
 If we use theano we need to use 3,64,64
'''
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))
# Adding a second convolutional layer
classifier.add(Convolution2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
'''image augmentation to avoid overfitting
   it allows us to enrich our dataset without adding any new images but creating
   new from the same datasets by introducing some distortions in our original training set'''
from keras.preprocessing.image import ImageDataGenerator
   
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
''' rescale makes sure that pixel values are converted from 0-1'''
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

#Making new prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,0)##it means that new dimension will be zero
'''the 4th dimension we need to add is the batch'''
result=classifier.predict(test_image)
training_set.class_indices
##

#Part-3 Tuning the CNN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

def build_classifier(optimizer,filters,units):
    classifier = Sequential()
    classifier.add(Convolution2D(filters,(3,3),input_shape=(64,64,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Convolution2D(filters,(3,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = units,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
    

classifier = KerasClassifier(build_fn = build_classifier,sk_params)
parameters = {'optimizer':['adams','rmsprop'],'filters':[48,64],'units':[180,256],
              'epochs':[10],'batch_size':[25]}  
from keras.preprocessing.image import ImageDataGenerator
   
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
''' rescale makes sure that pixel values are converted from 0-1'''
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring=['accuracy'],cv=5,
                           n_jobs=-1)

