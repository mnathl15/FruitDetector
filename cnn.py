from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
import h5py

classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128,activation="relu"))
classifier.add(Dense(output_dim = 4,activation="softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

classifier.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)





training_set = train_datagen.flow_from_directory(
                            'dataset/training',
                            target_size=(64, 64),
                            batch_size=64,
                            class_mode='categorical')

test_set= test_datagen.flow_from_directory(
                            'dataset/testing',
                            target_size=(64, 64),
                            batch_size=64,
                            class_mode='categorical')

classifier.fit_generator(training_set,
        steps_per_epoch=30,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=5)


model_json = classifier.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model.h5")
print('SAVED')



import numpy as np
from keras.preprocessing import image

test_image = image.load_img('test_apple2.jpg',target_size = (64,64))
test_image = image.img_to_array(test_image) #makes 64x64 to a 64x64x3
test_image = np.expand_dims(test_image,axis = 0) #classifier needs a 4d array in which last dimension is num of batches

#training_set.class_indices
result = classifier.predict(test_image)
print(result)
print(training_set.class_indices)

