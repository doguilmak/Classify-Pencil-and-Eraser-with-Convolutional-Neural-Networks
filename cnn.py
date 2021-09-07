"""
Created on Wed Aug 25 13:31:16 2021

@author: doguilmak
http://www.cs.cmu.edu/~aharley/vis/conv/flat.html
http://www.cs.cmu.edu/~aharley/vis/

Annotation: For improve the model, you can add more different eraser and 
pencil photos to the test_set and training_set files.

"""
#%%
# 1. Importing Libraries

import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

#%%
# PART 1 - CNN

# Initialization
classifier = Sequential()
start = time.time()

# Step 1 - First Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))  
# input_shape = (64, 64, 3) size of 64x64 pictures with RGB colors (3 primary colors).

# Adım 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adım 3 - Second Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adım 4 - Flattening
classifier.add(Flatten())

# Adım 5 - Artificial Neural Network
classifier.add(Dense(output_dim = 128, activation = 'relu'))  # Gives 128bit output
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))  # Returns a value of 1 or 0 (it can be said that a binomial determination is made to determine the male and female class)

# CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  # Gives best probability
classifier.summary()

#%%
# PART 2 - CNN and Pictures

from keras.preprocessing.image import ImageDataGenerator
## ImadeDataGenerator library for pictures.
## The difference from normal picture readings is that it evaluates the pictures one by one, not all at once and helps the RAM to work in a healthy way.

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
## shear_range = Side bends
## zoom_range = Zoom

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Train data
training_set = train_datagen.flow_from_directory('data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')
## target_size= 64x64 size pictures for scan.
## class_mode= Binary set

# Data is tested
test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')
# Train Artificial Neural Network
classifier.fit_generator(training_set,
                         samples_per_epoch = 4000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)
## training_set, = The training set that has been read is added
## nb_epoch = Number of epoch
## samples_per_epoch = How many samples will be made in each epoch
## validation_data = Data to be validated is added
## nb_val_samples = For determine the number of data to be validated 


#%%
# PART 3 - Prediction Eraser of Pencil

import numpy as np
import pandas as pd

test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)
#pred = list(map(round,pred))
## Filter predictions
pred[pred > .5] = 1
pred[pred <= .5] = 0
print('Prediction successful.')
#labels = (training_set.class_indices)


#%%
# PART 4 - Creating Confusion Matrix 

test_labels = []

for i in range(0, int(30)):  # 30 samples
    test_labels.extend(np.array(test_set[i][1]))
print('Test Labels(test_labels):\n')
print(test_labels)

#labels = (training_set.class_indices)
'''
idx = []  
for i in test_set:
    ixx = (test_set.batch_index - 1) * test_set.batch_size
    ixx = test_set.filenames[ixx : ixx + test_set.batch_size]
    idx.append(ixx)
    print(i)
    print(idx)
'''

# How each file was estimated and compared with real data is shown:
dosyaisimleri = test_set.filenames  
#abc = test_set.
#print(idx)
#test_labels = test_set.
sonuc = pd.DataFrame()
sonuc['dosyaisimleri']= dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels   

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
print ("Confusion Matrix:\n", cm)

end = time.time()
cal_time = end - start
print("\nTook {} seconds to classificate objects.".format(cal_time))
