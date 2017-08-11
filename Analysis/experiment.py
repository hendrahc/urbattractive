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

from keras.wrappers.scikit_learn import KerasClassifier
import math

def create_feature_extractor(weight_file='../../CNN/PredefinedModels/vgg_places_keras.h5'):
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
    inputs = img_input

    # Create model
    model = Model(inputs, x, name='extractor')
    model.compile(loss='binary_crossentropy',optimizer="SGD")

    #Load weight
    model.load_weights(weight_file,by_name=True)

    return model

def create_predictor(droprate1 = 0,droprate2 = 0, vgg=False, weight_file='../../CNN/PredefinedModels/vgg_places_keras.h5'):
    input_shape = (25088,)
    feat_input = Input(shape=input_shape)

    x = Dense(4096, activation='relu', name='fc6', trainable=True)(feat_input)
    x = Dropout(droprate1, name="dropout_1")(x)
    x = Dense(4096, activation='relu', name='fc7', trainable=True)(x)
    x = Dropout(droprate2, name="dropout_2")(x)
    if (vgg) :
        x = Dense(205, activation='softmax', name='fc8')(x)
    else:
        x = Dense(4, activation='sigmoid', name='predictor', trainable=True)(x)

    # Create model
    model = Model(feat_input, x, name='predictor')

    model.compile(loss='binary_crossentropy',
                  optimizer="SGD",
                  # metrics=[class_accuracy]
                  metrics=[]
                  )

    # Load weight
    model.load_weights(weight_file, by_name=True)

    return model


def extract_features(X,extractor= create_feature_extractor(),out_file="CNNModels/features.csv", use_generator=False, epoch=5):

    if use_generator:
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            channel_shift_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow(
            X,
            batch_size=10
        )

        feats = extractor.predict_generator(train_generator,steps=1)
    else:
        feats = extractor.predict(X)


    if (out_file):
        np.savetxt(out_file,feats)

    return feats

def train_predictor(model,F_train,Y_train,F_val,Y_val,epochs = 5,batch_size = 10,lr=0.01,decay=0.00001,callbacks_list=[]):

    nb_train_samples = F_train.shape[0]
    nb_validation_samples = F_val.shape[0]

    optim = keras.optimizers.SGD(lr=lr, momentum=0.9, decay=decay, nesterov=True);

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  # metrics=[class_accuracy]
                  metrics=[]
                  )

    for ep in range(1,epochs+1):
        model.fit(
            x=F_train,
            y=Y_train,
            batch_size=batch_size,
            epochs=1,
            callbacks=callbacks_list,
            validation_data= (F_val,Y_val),
            shuffle=True
        )

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

def load_dataset(path,ref_file, width, vals = []):
    ref = pd.read_csv(ref_file)
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    for index,row in ref.iterrows():
        filename = path + row["img_path"]
        img_id = row["img_id"]
        is_val = (img_id in vals)

        img = ""
        if is_val:
            img = image.load_img(filename, target_size=(224, 224))
        else:
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
            y = [1, 1, 0, 0]

        if is_val:
            X_val.append(x)
            Y_val.append(y)
        else:
            X_train.append(x)
            Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    return [X_train, Y_train, X_val, Y_val]

def load_exp_view(path, ref_file, width):
    ref = pd.read_csv(ref_file)
    X_train = []
    Y_train = []
    for index,row in ref.iterrows():
        filename = path + row["img_name"]
        img = image.load_img(filename, target_size=(width, width))
        x = image.img_to_array(img)
        cls = row["attractiveness"]
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
            y = [1, 1, 0, 0]

        X_train.append(x)
        Y_train.append(y)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return [X_train,Y_train]

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
    epochs = 10
    batch_size = 10

    optim = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.000001, nesterov=True);

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  #metrics=[class_accuracy]
                  metrics=[]
                  )

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        channel_shift_range = 0.2,
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
        validation_steps=nb_validation_samples // batch_size,
        callbacks = callbacks_list)
    return model

def start_model():
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_predictor()

    last = model.layers[-2].output
    #x = Dense(4096, activation='relu', name='fc7new', trainable=True)(last)
    x = Dense(4, activation='sigmoid', name='predictor', trainable=True)(last)
    model = Model(model.input, x, name='newModel')

    return model

def testModel(model,X_val,Y_val):
    preds = model.predict(X_val,batch_size=10)
    preds = binarize_result(preds)
    [accuracy, rmse] = get_evaluation(preds,Y_val)
    print("accuracy = "+str(accuracy))
    return preds

def get_places_ref(ref_path = '../../CNN/PredefinedModels/categoryIndex_places205.csv'):
    df_cat = pd.read_csv(ref_path,delimiter=" ")
    res = {}
    for idx,row in df_cat.iterrows():
        ctg = row["category"].split("/")[2]
        res[str(row["id"])] = ctg
    return res

def decode_scene(preds,reff=get_places_ref()):
    sorted = np.flipud(preds.argsort())
    for i in range(0,5):
        ct = sorted[i]
        print("["+str(preds[ct])+"] "+reff[str(ct)])

def decode_class(bins):
    if(bins[0]==0):
        return 1
    elif (bins[1]==0):
        return 2
    elif (bins[2] == 0):
        return 3
    elif (bins[3] == 0):
        return 4
    elif (bins[3] == 1):
        return 5
    return 3

def get_evaluation(Y_pred,Y_true):
    Y_pred = binarize_result(Y_pred)
    n = Y_pred.shape[0]
    correct = 0
    sum_error = 0.0
    for i in range(0,n):
        y_pred = decode_class(Y_pred[i])
        y_true = decode_class(Y_true[i])

        if(y_pred == y_true):
            correct = correct+1

        sum_error = sum_error + (y_true - y_pred)**2

    accuracy = float(correct)/float(n)
    rmse = math.sqrt(sum_error/float(n))
    return [accuracy, rmse]

def crop_im224(Xbig,xx,yy):
    img = image.array_to_img(Xbig)
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

def convert_to_binary(pred):
    res = np.zeros(pred.shape).astype(int)
    res[pred>0.5] = 1
    return res

def binarize_result(preds):
    p = np.zeros(preds.shape).astype(int)
    for i in range(0,preds.shape[0]):
        p[i] = convert_to_binary(preds[i])
    return p

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        print(self.losses)


def run_training(name):
    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    val_list = "CrowdData/val_list.csv"
    vals = pd.read_csv(val_list)["img_id"].values.tolist()
    [X_train, Y_train, X_val, Y_val] = load_dataset(path, ref, 224, vals)
    #[X_train, Y_train, X_val, Y_val] = load_dataset(path, ref, 400, vals)

    X_train = preprocess_dataset(X_train)
    X_val = preprocess_dataset(X_val)

    #use expansion dataset
    exp_path = "../../DATA/Expansion_view/"
    exp_ref = "Expansion/attr_exp_view.csv"
    exp_ref_same = "Expansion/attr_exp_view_same.csv"
    #[X_train, Y_train] = load_exp_view(exp_path, exp_ref, 224)
    #X_train = preprocess_dataset(X_train)

    #activate cropping
    #[X_train, Y_train] = get_crops(X_train, Y_train)
    #[X_val, Y_val] = get_crops(X_val, Y_val)

    model = start_model()

    #previous training load
    #model.load_weights("??")

    checkpath = "checks_"+name+"_{epoch:02d}_loss_{loss:.2f}_{val_loss:.2f}.h5"
    checkp = keras.callbacks.ModelCheckpoint(checkpath, monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=True, mode='auto', period=1)

    losslog = LossHistory()
    callbacks_list = [checkp]

    model = train_model(model, X_train, Y_train, X_val, Y_val,callbacks_list)

    #save_model(model,"../../CNN/Models/"+name+".json","../../CNN/Models/"+name+".h5")

def collect_log(logname, modelfiles = []):
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"

    val_list = "CrowdData/val_list.csv"
    vals = pd.read_csv(val_list)["img_id"].values.tolist()
    [X_train, Y_train, X_val, Y_val] = load_dataset(path, ref, 224, vals)

    X_train = preprocess_dataset(X_train)
    X_val = preprocess_dataset(X_val)

    df_log = pd.DataFrame(columns=["modelname", "acc_train", "acc_val", "rmse_train", "rmse_val"])

    if (modelfiles == []):
        modelfiles = [x for x in os.listdir(".") if x.endswith('.h5')]
    for modelfile in modelfiles:
        print("checking "+modelfile)
        newlog = {}
        newlog["modelname"] = modelfile
        model = start_model()
        model.load_weights(modelfile)
        model.compile(loss='binary_crossentropy', optimizer="SGD", metrics=[])
        preds_val = model.predict(X_val)
        eval_val = get_evaluation(preds_val,Y_val)
        newlog["acc_val"] = eval_val[0]
        newlog["rmse_val"] = eval_val[1]
        print(modelfile + "|" +str(eval_val[0]) +"|"+ str(eval_val[1]))
        preds_train = model.predict(X_train)
        eval_train = get_evaluation(preds_train, Y_train)
        newlog["acc_train"] = eval_train[0]
        newlog["rmse_train"] = eval_train[1]
        print(modelfile + "|" + str(eval_train[0]) + "|" + str(eval_train[1]))

        df_log = df_log.append(newlog, ignore_index=True)
    df_log.to_csv(logname, sep=",")

def prepare_feats(F_loc = ""):
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    def_F_loc = "../../FEATS/F_all.txt"

    if(F_loc == ""):
        [X_train, Y_train, X_val, Y_val] = load_dataset(path, ref, 224)
        X_train = preprocess_dataset(X_train)
        extractor = create_feature_extractor()
        F_all = extractor.predict(X_train)
        F_loc = def_F_loc
        np.savetxt(F_loc)

    F_all = np.loadtxt(F_loc)
    val_list = "CrowdData/val_list.csv"
    vals = pd.read_csv(val_list)["img_id"].values.tolist()

    df_ref = pd.read_csv(ref)

    train_idx = df_ref[~df_ref["img_id"].isin(vals)].index.values
    val_idx = df_ref[df_ref["img_id"].isin(vals)].index.values

    F_train = F_all[train_idx]
    F_val = F_all[val_idx]

    np.savetxt("../../FEATS/F_train.txt",F_train)
    np.savetxt("../../FEATS/F_val.txt", F_val)




def run():
    prepare_feats(F_loc="../../FEATS/F_all.txt")

#run()