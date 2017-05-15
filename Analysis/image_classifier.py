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

from keras import metrics
from keras import optimizers

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
    x = Dense(4096, activation='relu', name='fc7', trainable=True)(x)
    x = Dense(205, activation='softmax', name='fc8')(x)

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16')
    return model

def load_PLACES_weight(model,weights_path):
    model.load_weights(weights_path)
    return model

def preprocess_image(x):
    x[0], x[1], x[2] = x[2].transpose() - 105.487823486, x[1].transpose() - 113.741088867, x[0].transpose() - 116.060394287
    return x

def preprocess_dataset(dat):
    n = dat.shape[0]
    for i in range(0,n):
        x = dat[i]
        x[0], x[1], x[2] = x[2].transpose() - 105.487823486, x[1].transpose() - 113.741088867, x[0].transpose() - 116.060394287
        dat[i] = x
    return dat

def save_model(model,out_file,weight_file,layer_name=""):
    model_json = model.to_json()
    with open(out_file, "w") as json_file:
        json_file.write(model_json)
    json_file.close()
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
            y = [0, 0, 0, 0]

        '''
        if cls == 1:
            y = [1, 0, 0, 0,0]
        elif cls == 2:
            y = [0, 1, 0, 0, 0]
        elif cls == 3:
            y = [0, 0, 1, 0, 0]
        elif cls == 4:
            y = [0, 0, 0, 1, 0]
        elif cls == 5:
            y = [0, 0, 0, 0, 1]
        else:
            continue
        '''

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return [X,Y]

def predict_attractiveness(model,img_path):
    x = read_img(img_path)
    x = preprocess_image(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


def class_accuracy(y_true,y_pred):
    return np.array_equal(y_true,y_pred)


def train_model(model,X_train,Y_train,X_val,Y_val):
    # dimensions of our images.
    img_width, img_height = 224, 224
    nb_train_samples = X_train.shape[0]
    nb_validation_samples = X_val.shape[0]
    epochs = 100
    batch_size = 1

    optim = keras.optimizers.SGD(lr=10, momentum=0.0, decay=0.8, nesterov=False);

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy',metrics.categorical_accuracy])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

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
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model()
    model = load_PLACES_weight(model, WEIGHTS_PATH)

    last = model.layers[-1].output
    #x = Dense(4096, activation='relu', name='fc7new', trainable=True)(last)
    x = Dense(4, activation='sigmoid', name='predictor', trainable=True)(last)
    model = Model(model.input, x, name='newModel')
    return model

def testModel(model,X_val,Y_val):
    preds = model.predict(X_val,batch_size=10)
    preds = binarize_result(preds)
    accuracy = get_accuracy(preds,Y_val)
    print("accuracy = "+str(accuracy))
    return preds

def get_accuracy(Y_pred,Y_true):
    n = Y_pred.shape[0]
    correct = 0
    for i in range(0,n):
        if(np.array_equal(Y_pred[i],Y_true[i])):
            correct = correct+1
    return float(correct)/float(n)

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
    x = preprocess_image(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    decode_scene(preds,reff)
    return preds

def predict_scenes():
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    dat = pd.read_csv(ref)
    reff = get_places_ref()
    [X,Y] = load_dataset(path, ref)
    X = preprocess_dataset(X)

    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model()
    model = load_PLACES_weight(model, WEIGHTS_PATH)

    preds = model.predict(X)

    df_scene = pd.DataFrame(
        columns=["img_id", "scene1", "scene2", "scene3", "scene4", "scene5"])

    for i in range(0,preds.shape[0]):
        scene = {}
        pred_i = preds[i]
        scene["img_id"] = dat["img_id"][i]
        sorted = np.flip(pred_i.argsort(), 0)
        for j in range(1, 6):
            ct = sorted[j-1]
            scene["scene"+str(j)] = "'"+reff[str(ct)]+"'"
        df_scene = df_scene.append(scene, ignore_index=True)

        df_scene["img_id"] = df_scene["img_id"].astype(int)
    df_scene.to_csv("Data/SceneFeatures.csv")



def experiment():
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    [X,Y] = load_dataset(path, ref)
    X = preprocess_dataset(X)

    # split for training and validation
    n = X.shape[0]
    n_fold = 5
    fold_size = int(n/n_fold)

    for fold in range(0,n_fold):
        forval = [i for i in range(fold*fold_size, (fold+1)*fold_size)]
        fortrain = [i for i in range(0, n) if i not in forval]
        X_train = X[fortrain]
        Y_train = Y[fortrain]
        X_val = X[forval]
        Y_val = Y[forval]

        model = start_model()
        model = train_model(model, X_train, Y_train, X_val, Y_val)

def convert_to_binary(pred):
    res = np.zeros(pred.shape).astype(int)
    res[pred>0.5] = 1
    return res

def binarize_result(preds):
    for i in range(0,preds.shape[0]):
        preds[i] = convert_to_binary(preds[i])
    return preds


def run_training(name):
    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    [X,Y] = load_dataset(path,ref)

    n = X.shape[0]
    n_fold = 5
    fold_size = int(n / n_fold -1)

    X = preprocess_dataset(X)
    fold = 0
    forval = [i for i in range(fold * fold_size, (fold + 1) * fold_size)]
    fortrain = [i for i in range(0, n) if i not in forval]
    X_train = X[fortrain]
    Y_train = Y[fortrain]
    X_val = X[forval]
    Y_val = Y[forval]

    model = start_model()
    model = train_model(model, X_train, Y_train, X_val, Y_val)

    save_model(model,"../../CNN/Models/"+name+".json","../../CNN/Models/"+name+".h5")

def test_scene_prediction():
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model()

    model = load_PLACES_weight(model, WEIGHTS_PATH)

    img_path = '../Website/crowdsourcing/public/images/PILOT/GSV_PILOT_2_2.jpg'
    reff = get_places_ref()
    tes = classify_scene(model,reff,img_path)
    return model


def tesTrial():
    model = load_model("../../CNN/Models/trial_naif.json", "../../CNN/Models/trial_naif.h5")

    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    [X,Y] = load_dataset(path,ref)
    X = preprocess_dataset(X)
    X_tes = X[0:10]
    Y_tes = Y[0:10]
    preds = model.predict(X_tes, batch_size=10)

    return [model,X,Y,X_tes,Y_tes,preds]

def tes_training_work():
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    [X, Y] = load_dataset(path, ref)
    X = preprocess_dataset(X)

    X_train = X[0:4]
    Y_train = Y[0:4]
    Y_train[0] = np.array([1,1,0,0])
    Y_train[1] = np.array([0, 1, 1, 0])
    Y_train[2] = np.array([0, 0, 1, 1])
    Y_train[3] = np.array([0, 0, 0, 1])

    model = start_model()
    model = train_model(model, X_train, Y_train, X_train, Y_train)

    save_model(model, "../../CNN/Models/overfit.json", "../../CNN/Models/overfit.h5")

#run_training("trial_naif")
#tes_training_work()

#[model,X,Y,X_tes,Y_tes,preds] = tesTrial()
#test_prediction()
predict_scenes()

