import urllib
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import os.path

width = 600
height = 400
size = str(width) + "x" + str(height)
pitch = -0.76

def generate_gsv_url(size,lat,long,heading,pitch):
    location = str(lat) + "," + str(long)
    url = "https://maps.googleapis.com/maps/api/streetview?size=" + size + "&location=" + location + "&heading=" + str(
        heading) + "&pitch=" + str(pitch) + "&key=AIzaSyDeww92hY7OZDVGFyE7u5wHKXInVBmujHg"
    #print(url)
    return url


def image_valid(lat, long):
    heading = 0
    url = generate_gsv_url(size,lat,long,heading,pitch)
    urllib.request.urlretrieve(url, "test.jpg")
    statinfo = os.stat("test.jpg")
    if (statinfo.st_size < 7000):
        return 0
    return 1


def download_image(lat, long, heading, filename):
    url = generate_gsv_url(size,lat,long,heading,pitch)
    urllib.request.urlretrieve(url, filename)

#def get_nearby_image(lat,long):

#def get_other_views(lat,long, init_heading, known_values):

def get_hist_feat(img_file):
    hist_feat = np.array([])
    img = cv2.imread(img_file)
    #plt.figure()
    color = ('g','b','r')
    for i,col in enumerate(color):
        hist = cv2.calcHist([img], [i],None,[16],[0,16])
        hist_feat = np.append(hist_feat,hist)
        #plt.plot(hist)
    #plt.show()
    return hist_feat

def dist_img(img1,img2):
    hist1 = get_hist_feat(img1)
    hist2 = get_hist_feat(img2)
    return dist_hist(hist1, hist2)

def dist_hist(hist1,hist2):
    dis = plt.mlab.dist(hist1, hist2)
    return dis

def read_ref(img_data_f,loc_data_f,loc_im_f):
    df_img = pd.read_csv(img_data_f)
    df_loc = pd.read_csv(loc_data_f)
    df_loc_im = pd.read_csv(loc_im_f)
    views_im = {}
    for index, row in df_loc_im.iterrows():
        views_im[row["loc_id"]] = {"img1": row["img1"], "img2": row["img2"], "img3": row["img3"], "img4": row["img4"]}
    return [df_img,df_loc,views_im]

def expand_view(id,df_img,df_loc,loc_im):
    expand_view_dir = "../../DATA/Expansion_view/"
    headings = df_img[df_img["loc_id"]==id]["heading"].values

    #get coordinate
    [lat,long] = df_loc[df_loc["loc_id"]==id].values[0][[1,2]]
    for i in [0,1,2,3]:
        current_img = "img"+str(i+1)
        right_img = "img" + str((i+1)%4+1)

        c_heading = headings[i]
        #check to the right
        for h_d in [i*10 for i in range(0,9)]:
            head = (c_heading + h_d)%360
            fname = expand_view_dir+"EXPV_"+str(id)+"_"+str(head)+".jpg"
            if not(os.path.isfile(fname)):
                download_image(lat, long, head, fname)
            else:
                print("File "+fname+" is already exists")

            #labeling


def hist_compare(id,df_img):
    headings = df_img[df_img["loc_id"] == id]["heading"].values
    imgs =  df_img[df_img["loc_id"] == id]["filepath"].values

    ds = {}
    imgpath = {}
    for h in [0,1,2,3]:
        ds[h] = []
        #imgpath[h] = "../../DATA/"+imgs[h]
        imgpath[h] = "../../DATA/Expansion_view/EXPV_"+str(id)+"_"+str(headings[h])+".jpg"

    ax = [(i*10+headings[0])%360 for i in range(0,36)]

    for v in ax:
        target = "../../DATA/Expansion_view/EXPV_"+str(id)+"_"+str(v)+".jpg"
        for h in [0, 1, 2, 3]:
            d = dist_img(imgpath[h],target)
            ds[h].append(d)

    plt.figure()
    for h in [0, 1, 2, 3]:
        plt.plot(ds[h])
    plt.xticks([i for i in range(0,36)],ax)
    plt.show()





img_file = "../../DATA/PILOT/GSV_PILOT_515_2.jpg"
h = get_hist_feat(img_file)

dist_img("../../DATA/PILOT/GSV_PILOT_337_1.jpg","../../DATA/PILOT/GSV_PILOT_335_1.jpg")
dist_img('../../DATA/Expansion_view/EXPV_8_209.jpg',"../../DATA/PILOT/GSV_PILOT_8_3.jpg")





img_data_filename = "Data/images.csv"
loc_im_filename = "Data/loc_im.csv"
loc_data_filename = "Data/locations.csv"
[df_img,df_loc,loc_im] = read_ref(img_data_filename,loc_data_filename,loc_im_filename)

for lid in df_loc["loc_id"].values:
    expand_view(lid, df_img, df_loc, loc_im)