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

from keras.models import Model, Sequential
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

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
import h5py


def create_basic_model():
    input_shape = (3,224,224)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6', trainable=False)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(205, activation='softmax', name='fc8')(x)

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16')
    return model

def create_basic_model2():
    input_shape = (3,224, 224)
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(205, name='fc8'))

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

def load_PLACES_weight(model,weights_path):
    model.load_weights(weights_path)
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
    #WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    #WEIGHTS_PATH_NO_TOP = '../../CNN/PredefinedModels/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places.h5'
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

def testModel(model,X_val,Y_val):
    preds = model.predict(X_val,batch_size=1)
    preds = preds.astype(int)
    n = preds.shape[0]
    correct = 0
    for i in range(0,n):
        if(np.array_equal(preds[i],Y_val[i])):
            correct = correct+1
    return correct/n


def convert_weight(h5_file = '../../CNN/PredefinedModels/vgg_places.h5',out_file = '../../CNN/PredefinedModels/vgg_places_keras.h5'):
    res = h5py.File(out_file,'r+')
    f = h5py.File(h5_file,'r')
    ff = f[u'data'].values()
    at = np.array(f.keys()).astype("|S12")
    for dat in ff:
        for dts in dat.values():
            nm = dts.name.split("/")[2]
            idx = dts.name.split("/")[3]
            dtsname = "/" + nm + "/" + nm+"/"
            if(idx=="0"):
                dtsname = dtsname +"kernel:0"
            elif(idx=="1"):
                dtsname = dtsname +"bias:0"
            print (dts.name+" => "+dtsname)
            del res[dtsname]
            res[dtsname] = dts.value.transpose()
    res.close()
    return res

def get_places_ref(ref_path = '../../CNN/PredefinedModels/categoryIndex_places205.csv'):
    df_cat = pd.read_csv(ref_path,delimiter=" ")
    res = {}
    for idx,row in df_cat.iterrows():
        ctg = row["category"].split("/")[2]
        res[str(row["id"])] = ctg
    return res

def decode_scene(preds,reff):
    sorted = np.flip(preds.argsort(),0)
    for i in range(0,5):
        ct = sorted[i]
        print("["+str(preds[ct])+"] "+reff[str(ct)])


def classify_scene(model,reff,img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x[0],x[1],x[2] = x[2].transpose()-105.487823486,x[1].transpose()-113.741088867,x[0].transpose()-116.060394287
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    decode_scene(preds,reff)
    return preds

def run_training(name):
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model()
    model = load_PLACES_weight(model,WEIGHTS_PATH)

    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    [X_train,Y_train,X_val,Y_val] = load_dataset(path,ref)

    model = start_model()
    model = train_model(model,X_train,Y_train,X_val,Y_val)

    save_model(model,"CNNModels/"+name,"CNNModels/"+name)

def test_prediction():
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model2()
    model = load_PLACES_weight(model, WEIGHTS_PATH)

    img_path = '../Website/crowdsourcing/public/images/PILOT/GSV_PILOT_2_2.jpg'
    reff = get_places_ref()
    tes = classify_scene(model,reff,img_path)

'''
img_path = '../Website/crowdsourcing/public/images/PILOT/GSV_PILOT_93_2.jpg'
preds = predict_VGG(model,img_path)
print('Predicted:', decode_predictions(preds))
mymodel = load_model("CNNModels/trial2.json","CNNModels/trial2.h5")
score = mymodel.evaluate(X_val, Y_val, batch_size=1)
preds = mymodel.predict(X_val,batch_size=1)
'''

#run_training("trial1")
#test_prediction()

