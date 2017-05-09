import caffe
import numpy as np
import pandas as pd

caffe.set_mode_cpu()

model_def = "/data/hendra/Caffe/places205vgg/places205VGG.prototxt"
model_weights = "/data/hendra/Caffe/places205vgg/places205VGG.caffemodel"

def get_places_ref(ref_path = '/data/hendra/urbattractive/CNN/PredefinedModels/categoryIndex_places205.csv'):
    df_cat = pd.read_csv(ref_path,delimiter=" ")
    res = {}
    for idx,row in df_cat.iterrows():
        ctg = row["category"].split("/")[2]
        res[str(row["id"])] = ctg
    return res

def create_transformer():
    mu = np.array([105.487823486, 113.741088867, 116.060394287])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    return transformer

def decode_scene(preds,reff):
    sorted = np.flip(preds.argsort(),0)
    for i in range(0,5):
        ct = sorted[i]
        print("["+str(preds[ct])+"] "+reff[str(ct)])

def classify_scene(net, transformer,img_path):
    #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    image = caffe.io.load_image(img_path)
    transformed_image = transformer.preprocess('data', image)
    #tes = np.expand_dims(transformed_image.transpose(), axis=0)
    #dat = np.concatenate((tes,tes,tes,tes,tes,tes,tes,tes,tes,tes),axis=0)
    dat = transformed_image
    net.blobs['data'].data[...] = dat
    output = net.forward()
    preds = output['prob'][0]
    decode_scene(preds,reff)
    return preds

net = caffe.Net(model_def, model_weights,caffe.TEST)
reff = get_places_ref()
res = classify_scene(net,"/data/hendra/urbattractive/urbattractive/Website/crowdsourcing/public/images/PILOT/GSV_PILOT_35_3.jpg")
img_path = "/data/hendra/urbattractive/CNN/Testing/cornfield.jpg"