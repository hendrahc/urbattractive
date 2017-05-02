'''
import os
os.chdir("Analysis")
sys.path.append("")
from image_classifier import *
'''

import os

import keras
import numpy as np
import warnings
import pandas as pd

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.models import model_from_json

from keras.utils import plot_model


def create_basic_model():
    input_shape = (224,224,3)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1', trainable=False)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16')
    return model



def load_VGG_weight(model,weights_path):
    # load weights
    model.load_weights(weights_path)
    if K.backend() == 'theano':
        layer_utils.convert_all_kernels_in_model(model)

    if K.image_data_format() == 'channels_first':
        maxpool = model.get_layer(name='block5_pool')
        shape = maxpool.output_shape[1:]
        dense = model.get_layer(name='fc1')
        layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model

def predict_VGG(model,img_path):
    x = read_img(img_path)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

def save_model(model,out_file,weight_file,layer_name):
    model_json = model.to_json()
    with open(out_file, "w") as json_file:
        json_file.write(model_json)
    print("model saved")
    if(weight_file):
        model.save_weights(weight_file)
    print("weights saved")

def load_model(in_file,weight_file):
    json_file = open(in_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("model loaded")
    loaded_model = model_from_json(loaded_model_json)

    if (weight_file):
        loaded_model.load_weights(weight_file)
    print("weights loaded")

    loaded_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return loaded_model

def read_img(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    return x

def load_dataset(path,ref_file):
    ref = pd.read_csv(ref_file)
    X = []
    Y = []
    test = []
    for index,row in ref.iterrows():
        filename = row["img_path"]
        x = read_img(path+filename)
        cls = row["median"]
        y = []
        if cls==1:
            y = [0, 0, 0, 0]
        elif cls == 2:
            y = [1, 0, 0, 0]
        elif cls == 3:
            y = [1, 1, 0, 0]
        elif cls == 4:
            y = [1, 1, 1, 0]
        elif cls == 5:
            y = [1, 1, 1, 1]
        else:
            continue
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    #split for training and validation
    n = X.shape[0]
    #forval = [i for i in range(1,n) if i%10==4]
    forval = [i for i in range(0,56)]
    fortrain = [i for i in range(0,n) if i not in forval]
    X_train = X[fortrain]
    Y_train = Y[fortrain]
    X_val = X[forval]
    Y_val = Y[forval]

    return [X_train,Y_train,X_val,Y_val]

def predict_attractiveness(model,img_path):
    x = read_img(img_path)
    preds = model.predict(x)
    print(preds)
    return preds


def class_accuracy(y_true,y_pred):
    return np.array_equal(y_true,y_pred)


def train_model(model,X_train,Y_train,X_val,Y_val):
    # dimensions of our images.
    img_width, img_height = 224, 224
    nb_train_samples = 565
    nb_validation_samples = 565
    epochs = 20
    batch_size = 8

    optim = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.99, nesterov=False);

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=[class_accuracy])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(
        X_train, Y_train,
        batch_size=batch_size
    )

    test_generator = test_datagen.flow(
        X_val, Y_val,
        batch_size=batch_size
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=nb_validation_samples // batch_size)
    return model

def start_model():
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = '../../CNN/PredefinedModels/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    model = create_basic_model()
    model = load_VGG_weight(model,WEIGHTS_PATH)

    last = model.layers[-2].output
    x = Dense(4096, activation='relu', name='fc2new', trainable=True)(last)
    x = Dense(4, activation='softmax', name='predictor', trainable=True)(x)
    model = Model(model.input, x, name='newModel')

    #plot_model(model, to_file='CNNModels/model.png')

    #model_file = "CNNModels/model.json"
    #weight_file = "CNNModels/weight.h5"
    #save_model(model,model_file,weight_file)
    return model

def testModel(model,X_test,Y_test):
    preds = model.predict(X_test,batch_size=1)
    preds = preds.astype(int)
    n = preds.shape[0]
    correct = 0
    for i in range(0,n):
        if(np.array_equal(preds[i],Y_val[i])):
            correct = correct+1
    return correct/n


def run_training(name):
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model = create_basic_model()
    model = load_VGG_weight(model,WEIGHTS_PATH)

    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    [X_train,Y_train,X_val,Y_val] = load_dataset(path,ref)

    model = start_model()
    model = train_model(model,X_train,Y_train,X_val,Y_val)

    save_model(model,"CNNModels/"+name,"CNNModels/"+name)

'''
img_path = '../Website/crowdsourcing/public/images/PILOT/GSV_PILOT_93_2.jpg'
preds = predict_VGG(model,img_path)
print('Predicted:', decode_predictions(preds))
mymodel = load_model("CNNModels/trial2.json","CNNModels/trial2.h5")
score = mymodel.evaluate(X_val, Y_val, batch_size=1)
preds = mymodel.predict(X_val,batch_size=1)
'''

#run_training("trial1")

