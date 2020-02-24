#%%
import tensorflow as tf
import keras

from keras.applications.resnet_v2 import ResNet152V2

from keras.models import Sequential, Model
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine import training
from keras.layers import Add, add, merge, concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Flatten, Dense, InputLayer
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Input
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical, np_utils

from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

import os
import pandas as pd 
import cv2
import numpy as np
import pickle
import glob
from typing import Tuple, List
#%%
def make_img(image_folder):
    image_files = os.listdir(image_folder)
    tmp_img_array = []
    tmp_label_array = []
    for img in image_files:
        tmp_filename = os.path.join(image_folder, img)
        try:
            x = cv2.imread(tmp_filename)
            x = cv2.resize(x, (224, 224))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis = 2)
            # x = np.array(X, dtype="float32") / 255.0
            tmp_img_array.append(x)
        except:
            print(f'error at {tmp_filename}')
    return tmp_img_array
#%%
def normalize_X(X):
    X = np.array(X, dtype="float32") / 255.0
    print(type(X))
    return (X)
#%%
def make_pickle(data, name, input_path):
    fhandle = open(name, 'ab')
    pickle.dump(data, fhandle)
    fhandle.close()

def read_pickle(pickle_filename, input_path):
    fhandle = open(pickle_filename,"rb")
    pickle_file = pickle.load(fhandle)
    return pickle_file

def make_label_array(label_path, target):
    read_df = pd.read_excel(label_path)
    target = str(target)
    labels = read_df[target].values
    y_OHE = np_utils.to_categorical(labels)
    return y_OHE
#%%
def output_make(input_path, label_path, target):
    with tf.device('/cpu:0'):
        base_model = ResNet152V2(weights= None, include_top=False, input_shape= (224,224,1))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(5, activation= 'softmax')(x)
    
        resnet_model = Model(inputs = base_model.input, outputs = predictions)
        # resnet_hist = compile_and_train(resnet_model, num_epochs)
        resnet_model.load_weights('./PredictHouseholdIncome_pradeep4444.hdf5')
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        resnet_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        X = make_img(input_path)
        X = normalize_X(X)

        pickle_path = os.path.join(input_path, 'processed_pickles')
        make_pickle(X,'pickled_image_array', pickle_path)
        X = read_pickle('pickled_image_array', pickle_path)
        y_preds = resnet_model.predict(X)
        try:
            y_OHE = make_label_array(label_path, target)
            y_preds = resnet_aug.predict(X_test)
            matrix = confusion_matrix(y_OHE.argmax(axis=1), y_preds.argmax(axis=1))

            y_pred_bool = np.argmax(y_preds, axis=1)
            y_test_bool = np.argmax(y_OHE, axis=1)
            class_report = classification_report(y_test_bool, y_pred_bool)
            with open('output.txt','w') as op_handle:
                op_handle.write('raw predictions:\n')
                op_handle.write(str(y_preds))
                op_handle.write('\n')

                op_handle.write('confusion martix:\n')
                op_handle.write(str(matrix))
                op_handle.write('\n')

                op_handle.write('classification report:\n')
                op_handle.write(str(class_report))
                op_handle.write('\n')

        except:
            pass
#%%
input_path = './test_images'
label_path = './test_labels'
target = 'label'
output_make(input_path, label_path, target)