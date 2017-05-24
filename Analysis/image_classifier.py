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
import datetime
from PIL import Image

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
    x = Dense(4096, activation='relu', name='fc6', trainable=True)(x)
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
    x[0], x[1], x[2] = x[2].transpose() - 105, x[1].transpose() - 114, x[0].transpose() - 116
    return x

def preprocess_dataset(dat):
    n = dat.shape[0]
    for i in range(0,n):
        x = dat[i]
        x[0], x[1], x[2] = x[2].transpose() - 105, x[1].transpose() - 114, x[0].transpose() - 116
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

def load_dataset(path,ref_file,width):
    ref = pd.read_csv(ref_file)
    X = []
    Y = []
    test = []
    for index,row in ref.iterrows():
        filename = path + row["img_path"]
        img = image.load_img(filename, target_size=(width, width))
        x = image.img_to_array(img)
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
    return K.mean(K.equal(K.sum(K.abs(K.round(y_pred) - y_true),axis=-1),0))


def train_model(model,X_train,Y_train,X_val,Y_val,callbacks_list=[]):
    # dimensions of our images.
    img_width, img_height = 224, 224
    nb_train_samples = X_train.shape[0]
    nb_validation_samples = X_val.shape[0]
    epochs = 20
    batch_size = 5

    optim = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.000001, nesterov=True);

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=[class_accuracy])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator()
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)

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
        validation_steps=nb_validation_samples // batch_size,
        callbacks = callbacks_list)
    return model

def start_model():
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model()
    model = load_PLACES_weight(model, WEIGHTS_PATH)

    last = model.layers[-2].output
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


def crop_im224(Xbig,xx,yy):
    img = image.fromarray(Xbig, 'RGB')
    cropped = img.crop((xx,yy,xx+224,yy+224))
    return image.img_to_array(cropped)

def get_crops(X,Y):
    X_crop = []
    Y_crop = []
    n = Y.shape[0]

    for i in range(0,n):
        #crop_center
        X_crop.append(crop_im224(X[i],88,88))
        Y_crop.append(Y[i])

        # crop1
        X_crop.append(crop_im224(X[i], 0, 0))
        Y_crop.append(Y[i])

        # crop2
        X_crop.append(crop_im224(X[i], 0, 176))
        Y_crop.append(Y[i])

        # crop3
        X_crop.append(crop_im224(X[i], 176, 0))
        Y_crop.append(Y[i])

        # crop4
        X_crop.append(crop_im224(X[i], 176, 176))
        Y_crop.append(Y[i])

    X_crop = np.array(X_crop)
    Y_crop = np.array(Y_crop)
    return [X_crop, Y_crop]



def experiment(name):
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    [X,Y] = load_dataset(path, ref,224)
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
        print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        print("Training Fold "+str(fold))
        model = train_model(model, X_train, Y_train, X_val, Y_val)
        print("Training Fold " + str(fold)+" done..")
        save_model(model, "../../CNN/Models/" + name + ".json", "../../CNN/Models/" + name + "_"+str(fold+1)+".h5")


def convert_to_binary(pred):
    res = np.zeros(pred.shape).astype(int)
    res[pred>0.5] = 1
    return res

def binarize_result(preds):
    p = np.zeros(preds.shape).astype(int)
    for i in range(0,preds.shape[0]):
        p[i] = convert_to_binary(preds[i])
    return p


def run_training(name):
    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    #[X, Y] = load_dataset(path, ref, 224)
    [X,Y] = load_dataset(path,ref,400)

    n = X.shape[0]

    X = preprocess_dataset(X)
    val_ids = pd.read_csv("Data/val_img.csv")

    forval = val_ids["img_id"].values.tolist()
    fortrain = [i for i in range(0, n) if i not in forval]
    X_train = X[fortrain]
    Y_train = Y[fortrain]
    X_val = X[forval]
    Y_val = Y[forval]

    [X_train, Y_train] = get_crops(X_train, Y_train)
    [X_val, Y_val] = get_crops(X_val, Y_val)

    model = start_model()
    model = model.load_weights("??")

    checkpath = "../../CNN/Models/Checkpoints/checks_"+name+"_{epoch:02d}_acc_{class_accuracy:.2f}.h5"
    checkp = keras.callbacks.ModelCheckpoint(checkpath, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=True, mode='auto', period=10)

    callbacks_list = [checkp]

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
    [X,Y] = load_dataset(path,ref,224)
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

    save_model(model, "../../CNN/Models/overfit2.json", "../../CNN/Models/overfit2.h5")

#run_training("basic1")
#tes_training_work()

#[model,X,Y,X_tes,Y_tes,preds] = tesTrial()
#test_prediction()
#predict_scenes()
#experiment("basic_ori")
