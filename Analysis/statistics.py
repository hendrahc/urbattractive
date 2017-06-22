'''
import os
os.chdir("Analysis")
'''
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import *
import matplotlib.pyplot as plt
from shutil import copy
from sklearn.decomposition import FactorAnalysis

import scipy


def normalize(df):
    # normalization
    df["familiarity"] = df["familiarity"].map({'yes': 1, 'no': 0})
    df["friendliness"] = df["friendliness"].map({'yes': 1, 'no': 0})
    df["pleasure"] = df["pleasure"]
    df["arousal"] = df["arousal"]
    df["dominance"] = df["dominance"]

    df["img_id"] = df["img_id"].fillna(-99).astype(int)
    df["loc_id"] = df["loc_id"].fillna(-99).astype(int)

    return df

def norm_polar(df):
    df["attractiveness"] = df["attractiveness"] - 3
    df["familiarity"] = df["familiarity"].map({0: -2, 1: 2})
    df["uniqueness"] = df["uniqueness"] - 3
    df["friendliness"] = df["friendliness"].map({0: -2, 1: 2})
    df["pleasure"] = df["pleasure"]*2
    df["arousal"] = df["arousal"]*2
    df["dominance"] = df["dominance"]*2
    return df

def read_data(inp):
    df = pd.read_csv(inp)
    df = normalize(df)

    df_clean =  df[df["user_id"] >= 17]
    df_part1 = df_clean[df["part"].isin([0,1])]
    df_part2 = df_clean[df["part"]==2]
    return [df,df_part1,df_part2]

def read_ref(img_data_f,loc_im_f,loc_f):
    df_img = pd.read_csv(img_data_f)
    df_loc_im = pd.read_csv(loc_im_f)
    views_im = {}
    for index, row in df_loc_im.iterrows():
        views_im[row["loc_id"]] = {"img1": row["img1"], "img2": row["img2"], "img3": row["img3"], "img4": row["img4"]}
    df_loc = pd.read_csv(loc_f)
    return [df_img,views_im,df_loc]


def generate_loc_im(in_file,out_file):
    imm = pd.read_csv(in_file)
    loc_im = {}
    for idx,row in imm.iterrows():
        if row["loc_id"] > 0:
            loc = str(row["loc_id"])
            im = str(row["id"])
            if loc in loc_im:
                loc_im[loc] = loc_im[loc] + "," + im
            else:
                loc_im[loc] = "" + im

    outf = open(out_file, 'w')
    outf.write("loc_id,img1,img2,img3,img4\n")

    for r,v in loc_im.items():
        outf.write(r+","+v+"\n")

def corr_mat(dat):
    #compute correlation matrix
    df_attrib_scores =  dat[["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]]
    correl_mat = df_attrib_scores.corr()
    return correl_mat

def spearman_corr(dat):
    ats = ["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]
    corr,pval = scipy.stats.spearmanr(dat[ats])
    for i in range(0,7):
        line = ats[i]
        for j in range(0, 7):
            line = line + "|" + str(corr[i][j])
        print(line)


def save_corr_mat(cm,fname):
    cm.to_csv(fname)

def aggregate_data_part1(df,df_img):
    df_aggr = pd.DataFrame(columns=["img_id", "img_path", "num_user","mean","median","var","vote1","vote2","vote3","vote4","vote5"])
    for idx,row in df_img.iterrows():
        img_id = int(row["id"])
        df_filtered = df[df["img_id"]==img_id]
        values = df_filtered["attractiveness"].values

        if(df_filtered.shape[0]>0):
            newdat = {}
            newdat["img_id"] = img_id
            newdat["img_path"] = row["filepath"]
            newdat["num_user"] = df_filtered.shape[0]
            newdat["mean"] = np.nanmean(values)
            newdat["median"] = np.nanmedian(values)
            newdat["var"] = np.nanvar(values)

            #count votes
            for val in range(1,6):
                vote = df_filtered[df_filtered["attractiveness"]==val].shape[0]
                newdat["vote"+str(val)] = vote

            df_aggr = df_aggr.append(newdat,ignore_index=True)
    df_aggr["img_id"] = df_aggr["img_id"].astype(int)
    df_aggr["num_user"] = df_aggr["num_user"].astype(int)
    df_aggr["median"] = df_aggr["median"].astype(int)
    for val in range(1, 6):
        df_aggr["vote"+str(val)] = df_aggr["vote"+str(val)].astype(int)
    return df_aggr


def aggregate_attribute_part1(df,df_img):
    df_aggr_attr_part1 = pd.DataFrame(
        columns=["img_id", "num_user", "attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"])

    for idx, row in df_img.iterrows():
        img_id = int(row["id"])
        df_filtered = df[df["img_id"] == img_id]
        newdat = {}
        newdat["img_id"] = img_id
        newdat["num_user"] = df_filtered.shape[0]
        newdat["attractiveness"] = np.nanmedian(df_filtered["attractiveness"].values)
        newdat["familiarity"] = np.nanmedian(df_filtered["familiarity"].values)
        newdat["uniqueness"] = np.nanmedian(df_filtered["uniqueness"].values)
        newdat["friendliness"] = np.nanmedian(df_filtered["friendliness"].values)
        newdat["pleasure"] = np.nanmean(df_filtered["pleasure"].values)
        newdat["arousal"] = np.nanmean(df_filtered["arousal"].values)
        newdat["dominance"] = np.nanmean(df_filtered["dominance"].values)
        df_aggr_attr_part1 = df_aggr_attr_part1.append(newdat, ignore_index=True)
    df_aggr_attr_part1["img_id"] = df_aggr_attr_part1["img_id"].astype(int)
    df_aggr_attr_part1["num_user"] = df_aggr_attr_part1["num_user"].astype(int)
    df_aggr_attr_part1["attractiveness"] = df_aggr_attr_part1["attractiveness"].astype(int)
    df_aggr_attr_part1["familiarity"] = df_aggr_attr_part1["familiarity"].astype(int)
    df_aggr_attr_part1["uniqueness"] = df_aggr_attr_part1["uniqueness"].astype(int)
    df_aggr_attr_part1["friendliness"] = df_aggr_attr_part1["friendliness"].astype(int)

    return df_aggr_attr_part1



def aggregate_data_part2(df):
    df_aggr = pd.DataFrame(columns=["loc_id", "num_user","mean","median","var","vote1","vote2","vote3","vote4","vote5"])
    for loc_id in df["loc_id"].unique():
        df_filtered = df[df["loc_id"]==loc_id]
        values = df_filtered["attractiveness"].values

        if(df_filtered.shape[0]>0):
            newdat = {}
            newdat["loc_id"] = loc_id
            newdat["num_user"] = df_filtered.shape[0]
            newdat["mean"] = np.nanmean(values)
            newdat["median"] = np.nanmedian(values)
            newdat["var"] = np.nanvar(values)

            # count votes
            for val in range(1, 6):
                vote = df_filtered[df_filtered["attractiveness"] == val].shape[0]
                newdat["vote" + str(val)] = vote

            df_aggr = df_aggr.append(newdat,ignore_index=True)
    df_aggr["loc_id"] = df_aggr["loc_id"].astype(int)
    df_aggr["num_user"] = df_aggr["num_user"].astype(int)
    df_aggr["median"] = df_aggr["median"].astype(int)
    for val in range(1, 6):
        df_aggr["vote" + str(val)] = df_aggr["vote" + str(val)].astype(int)
    return df_aggr

def save_df(df,outname):
    df.to_csv(outname,sep=",")

def create_dataset_input(data,input_loc,output_loc):
    for idx, row in data.iterrows():
        cls = row["median"]
        if cls == 1:
            cls = 2
        elif cls == 5:
            cls = 4
        fl = row["img_path"]
        copy(input_loc+'/'+fl, output_loc+'/'+str(cls)+'/')

def summarize_data(df_aggr_part1, df_aggr_part2, loc_im):
    df_summary = pd.DataFrame(
        columns=["loc_id", "var1", "var2", "var3", "var4", "var_loc", "lab1", "lab2", "lab3", "lab4", "lab_loc"])
    for idx, row in df_aggr_part2.iterrows():
        newdat = {}
        loc_id = row["loc_id"]
        newdat["loc_id"] = loc_id
        filt_loc = df_aggr_part2[df_aggr_part2["loc_id"]==loc_id]
        newdat["var_loc"] = filt_loc["var"].values[0]
        newdat["lab_loc"] = filt_loc["median"].values[0]
        for i in range(1,5):
            img_id = loc_im[loc_id]["img"+str(i)]
            filt = df_aggr_part1[df_aggr_part1["img_id"]==img_id]
            newdat["var"+str(i)] = filt["var"].values[0]
            newdat["lab" + str(i)] = filt["median"].values[0]
        df_summary = df_summary.append(newdat, ignore_index=True)
    df_summary["loc_id"] = df_summary["loc_id"].astype(int)
    df_summary["lab_loc"] = df_summary["lab_loc"].astype(int)
    for i in range(1, 5):
        df_summary["lab"+str(i)] = df_summary["lab"+str(i)].astype(int)
    return df_summary


def label_tuning(df_summary):
    df_out = df_summary
    for idx, row in df_summary.iterrows():
        # tuning image labels
        for i in range(1, 5):
            if (row["var" + str(i)] > 1):
                lab_left = 1
                lab_right = 1

        #tuning overall attractiveness
        var_loc = row["var_loc"]
        lab_loc = row["lab_loc"]
        if(var_loc > 1):
            avg_lab = (row["lab1"]+row["lab2"]+row["lab3"]+row["lab4"])/4
            if(abs(lab_loc-avg_lab) >= 1):
                if lab_loc > avg_lab:
                    df_out[idx]["lab_loc"] = lab_loc - 1
                else:
                    df_out[idx]["lab_loc"] = lab_loc + 1


def get_attr_function(df):
    regr = linear_model.LinearRegression()
    X = df[["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]]
    X_train = df[["familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]]
    Y_train = df["attractiveness"]
    regr.fit(X_train,Y_train)

    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(X_train) - Y_train) ** 2))
    preds = regr.predict(X_train)
    preds = preds.round().astype(int)
    confmat = confusion_matrix(Y_train,preds)
    acc = accuracy_score(Y_train,preds)

def factor_loading(df):
    fa = FactorAnalysis(n_components=2)
    fa.fit(df[["attractiveness", "familiarity", "uniqueness", "friendliness", "pleasure", "arousal", "dominance"]])
    fa.components_

'''
#attractiveness function
df_scores = pd.DataFrame(columns=["user_id","loc_id","overall","img1","img2","img3","img4"])

user_ids = [7,8,9,10]

for user in user_ids:
    df_p2 = df_part2[df_part2["user_id"]==user]
    for index,row in df_p2.iterrows():
        attr = {}
        attr["user_id"] = user
        attr["loc_id"] = row["loc_id"]
        attr["overall"] = row["attractiveness"]
        ims = views_im[attr["loc_id"]]
        for im_idx in ["img1","img2","img3","img4"]:
            img_id = ims[im_idx]
            sel = df_part1[(df_part1["user_id"]==user) & (df_part1["img_id"]==img_id)]
            if(sel.shape[0] >= 1):
                attr[im_idx] = sel.iloc[0]["attractiveness"]
            else:
                attr[im_idx] = attr["overall"] #default
        df_scores = df_scores.append(attr,ignore_index=True)

regr = linear_model.LinearRegression()

X_train = df_scores[["img1","img2","img3","img4"]]
Y_train = df_scores["overall"]
regr.fit(X_train,Y_train)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_train) - Y_train) ** 2))

#MSE if average
print("Mean squared error: %.2f"
      % np.mean(((X_train["img1"]+X_train["img2"]+X_train["img3"]+X_train["img4"])/4 - Y_train) ** 2))


#ordered
df_scores_ordered = pd.DataFrame(columns=["s1","s2","s3","s4"])
for idx,row in df_scores.iterrows():
    arr_score = row[["img1","img2","img3","img4"]].values
    arr_score.sort()
    new_attr = {}
    new_attr["s1"] = arr_score[0]
    new_attr["s2"] = arr_score[1]
    new_attr["s3"] = arr_score[2]
    new_attr["s4"] = arr_score[3]
    df_scores_ordered = df_scores_ordered.append(new_attr,ignore_index=True)

X_train = df_scores_ordered[["s1","s2","s3","s4"]]
regr.fit(X_train,Y_train)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_train) - Y_train) ** 2))


#aggregate
df_aggr_part1 = df_part1.groupby(["img_id"])[["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]].mean()
df_aggr_part2 = df_part2.groupby(["img_id"])[["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]].mean()


df_scores_aggr = df_scores.groupby(["loc_id"])[["loc_id","overall","img1","img2","img3","img4"]].mean()
'''





### MAIN ###

#parameters
input_filename = "CrowdData/pilot_judgements.csv"
img_data_filename = "Data/images.csv"
loc_im_filename = "Data/loc_im.csv"
loc_filename = "Data/locations.csv"
corr_mat_filename = "CrowdData/corr_mat.csv"
aggr_part1_filename = "CrowdData/pilot_aggregates_part1.csv"
aggr_part2_filename = "CrowdData/pilot_aggregates_part2.csv"
input_image_loc = '../Website/crowdsourcing/public/images'
dataset_image_loc = 'InputImages/Training'
summary_filename = "CrowdData/summary_ori.csv"


def activities():
    #activities
    [df,df_part1,df_part2] = read_data(input_filename)
    [df_img,loc_im,df_loc] = read_ref(img_data_filename,loc_im_filename,loc_filename)

    df_aggr_part1 = aggregate_data_part1(df_part1,df_img)
    save_df(df_aggr_part1,aggr_part1_filename)

    df_aggr_part2 = aggregate_data_part2(df_part2)
    save_df(df_aggr_part2,aggr_part2_filename)

    create_dataset_input(df_aggr_part1,input_image_loc,dataset_image_loc)

    df_aggr_part1 = pd.read_csv(aggr_part1_filename)
    df_aggr_part2 = pd.read_csv(aggr_part2_filename)

    df_summary = summarize_data(df_aggr_part1, df_aggr_part2, loc_im)
    save_df(df_summary,summary_filename)