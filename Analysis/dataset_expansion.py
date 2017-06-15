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

def attr_f(input_dir,img_name,loc_im,df_img, df_vote):
    nameparts = img_name.split("_")
    loc_id =  int(nameparts[1])
    dir = int(nameparts[2].split(".")[0])
    i = 1
    img_id_1 = loc_im[loc_id]["img"+str(i)]
    img_1 = df_img[df_img["id"]==img_id_1]
    dir_1 = img_1["heading"].values[0]

    if((dir_1-dir)%90 == 0):
        pos_dir = ((dir-dir_1)//90)%4+1
        #print(str(dir)+" is the same as "+str(df_img[df_img["id"]==loc_im[loc_id]["img"+str(pos_dir)]]["heading"].values[0]))
        attr_dir = df_vote[df_vote["img_id"] == loc_im[loc_id]["img"+str(pos_dir)]]["median"].values[0]
        #print(str(dir)+" <=> "+str(attr_dir))
        return attr_dir
    else:
        pos_left_dir = ((dir-dir_1)//90)%4+1
        pos_right_dir = pos_left_dir%4+1

        dir_left = df_img[df_img["id"]==loc_im[loc_id]["img"+str(pos_left_dir)]]["heading"].values[0]
        dir_right = (dir_left+90)%360
        #print(str(dir) +" is between "+str(dir_left) + " and "+ str(dir_right))

        attr_left = df_vote[df_vote["img_id"] == loc_im[loc_id]["img"+str(pos_left_dir)]]["median"].values[0]
        attr_right = df_vote[df_vote["img_id"] == loc_im[loc_id]["img" + str(pos_right_dir)]]["median"].values[0]

        closeness_left =  ((dir_right-dir)%90) / 90
        closeness_right = ((dir-dir_left) % 90) / 90

        pred_attr =  closeness_left*attr_left + closeness_right*attr_right
        #print(str(dir_left)+":"+str(dir)+":"+str(dir_right)+" <=> "+str(attr_left)+":"+str(pred_attr)+":"+str(attr_right))
        return round(pred_attr)




def label_views(loc_im,df_img=pd.read_csv("Data/images.csv"),df_vote=pd.read_csv("CrowdData/pilot_aggregates_part1.csv"),input_dir="../../DATA/Expansion_view/",output_log="Expansion/attr_exp_view.csv"):
    df_expview = pd.DataFrame(columns=["loc_id","img_name","attractiveness"])

    files = [x for x in os.listdir(input_dir) if x.endswith('.jpg')]
    for loc_id in loc_im.keys():
        imgs = [x for x in files if x.split("_")[1]==str(loc_id)]
        for img_name in imgs:
            attr = attr_f(input_dir, img_name, loc_im,df_img, df_vote)
            newdat = {}
            newdat["loc_id"] = loc_id
            newdat["img_name"] = img_name
            newdat["attractiveness"] = attr
            df_expview = df_expview.append(newdat, ignore_index=True)
    df_expview["loc_id"] = df_expview["loc_id"].astype(int)
    df_expview["attractiveness"] = df_expview["attractiveness"].astype(int)
    df_expview.to_csv(output_log,sep=",")
    return df_expview


def label_views_linear(df_loc,loc_im,df_img=pd.read_csv("Data/images.csv"),df_vote=pd.read_csv("CrowdData/pilot_aggregates_part1.csv"),input_dir="../../DATA/Expansion_view/",output_log="Expansion/attr_exp_view_linear.csv"):
    df_expview_linear = pd.DataFrame(columns=["loc_id", "img_name", "attractiveness"])
    for loc_id in loc_im.keys():
        for pos in [0, 1, 2, 3]:
            pos_left = pos + 1
            pos_right = (pos + 1) % 4 + 1
            img_id_left = loc_im[loc_id]["img" + str(pos_left)]
            img_id_right = loc_im[loc_id]["img" + str(pos_right)]

            attr_left = df_vote[df_vote["img_id"] == img_id_left]["median"].values[0]
            attr_right = df_vote[df_vote["img_id"] == img_id_right]["median"].values[0]


            dir_left = df_img[df_img["id"] == img_id_left]["heading"].values[0]

            for k in [0, 1]:
                dir_new = (dir_left + (k+1)*30 )%360

                attr_pred = attr_left
                if k==1:
                    attr_pred = attr_right

                newdat = {}
                newdat["loc_id"] = loc_id
                newdat["img_name"] = input_dir + "EXPV_" + str(loc_id) + "_" + str(dir_new) + ".jpg"
                newdat["attractiveness"] = attr_pred
                df_expview_linear = df_expview_linear.append(newdat, ignore_index=True)

    df_expview_linear["loc_id"] = df_expview_linear["loc_id"].astype(int)
    df_expview_linear["attractiveness"] = df_expview_linear["attractiveness"].astype(int)
    df_expview_linear.to_csv(output_log, sep=",")
    return df_expview_linear



def label_views_same(df_loc,loc_im,df_img=pd.read_csv("Data/images.csv"),df_vote=pd.read_csv("CrowdData/pilot_aggregates_part1.csv"),input_dir="../../DATA/Expansion_view/",output_log="Expansion/attr_exp_view_same.csv"):
    df_expview_same = pd.DataFrame(columns=["loc_id", "img_name", "attractiveness"])
    for loc_id in loc_im.keys():
        for pos in [0,1,2,3]:
            pos_left = pos+1
            pos_right = (pos+1)%4+1
            img_id_left = loc_im[loc_id]["img"+str(pos_left)]
            img_id_right = loc_im[loc_id]["img" + str(pos_right)]

            attr_left = df_vote[df_vote["img_id"]==img_id_left]["median"].values[0]
            attr_right = df_vote[df_vote["img_id"] == img_id_right]["median"].values[0]

            if(attr_left == attr_right):
                dir_left = df_img[df_img["id"]==img_id_left]["heading"].values[0]
                dir_new = {}
                dir_new[1] = (dir_left+30)%365
                dir_new[2] = (dir_left + 60) % 365

                for k in [0,1]:
                    newdat = {}
                    newdat["loc_id"] = loc_id
                    newdat["img_name"] = input_dir+"EXPV_"+str(loc_id)+"_"+str(dir_new[k])+".jpg"
                    newdat["attractiveness"] = attr_left
                    df_expview_same = df_expview_same.append(newdat, ignore_index=True)

    df_expview_same["loc_id"] = df_expview_same["loc_id"].astype(int)
    df_expview_same["attractiveness"] = df_expview_same["attractiveness"].astype(int)
    df_expview_same.to_csv(output_log, sep=",")
    return df_expview_same





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

def expand_view(id,df_img,df_loc,loc_im,expand_view_dir = "../../DATA/Expansion_view/"):
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

#for lid in df_loc["loc_id"].values:
#    expand_view(lid, df_img, df_loc, loc_im)