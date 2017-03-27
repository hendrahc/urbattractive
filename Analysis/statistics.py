import pandas as pd
import numpy as np
from sklearn import linear_model

input_filename = "Data/prepilot.csv"
img_data_filename = "Data/images.csv"
loc_im_filename = "Data/loc_im.csv"


df = pd.read_csv(input_filename)
#normalization
df["familiarity"] = df["familiarity"].map({'yes': 5, 'no': 1})
df["friendliness"] = df["friendliness"].map({'yes': 5, 'no': 1})
df["pleasure"] = (df["pleasure"]*2)+3
df["arousal"] = (df["arousal"]*2)+3
df["dominance"] = (df["dominance"]*2)+3

df_part1 = df[df["part"].isin([0,1])]
df_part2 = df[df["part"]==2]



#compute correlation matrix
df_attrib_scores =  df[["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]]

correl_mat = df_attrib_scores.corr()


df_part1[["attractiveness","familiarity","uniqueness","friendliness","pleasure","arousal","dominance"]].corr()



#attractiveness function
df_images = pd.read_csv(img_data_filename)

df_loc_im = pd.read_csv(loc_im_filename)
views_im = {}
for index,row in df_loc_im.iterrows():
    views_im[row["loc_id"]] = {"img1": row["img1"],"img2": row["img2"], "img3": row["img3"], "img4": row["img4"]}

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
