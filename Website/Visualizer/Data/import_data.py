import json
import pandas as pd


def read_ref(img_data_f,loc_im_f, loc_f):
    df_img = pd.read_csv(img_data_f)
    df_loc_im = pd.read_csv(loc_im_f)
    df_loc = pd.read_csv(loc_f)
    views_im = {}
    for index, row in df_loc_im.iterrows():
        views_im[row["loc_id"]] = {"img1": row["img1"], "img2": row["img2"], "img3": row["img3"], "img4": row["img4"]}
    return [df_img,views_im,df_loc]

def import_judgements():
    img_data_filename = "../../../Analysis/Data/images.csv"
    loc_im_filename = "../../../Analysis/Data/loc_im.csv"
    loc_filename = "../../../Analysis/Data/locations.csv"
    [df_img,loc_im,df_loc] = read_ref(img_data_filename,loc_im_filename, loc_filename)

    aggr_part1_f = "../../../Analysis/CrowdData/pilot_aggregates_part1.csv"
    aggr_part2_f = "../../../Analysis/CrowdData/pilot_aggregates_part2.csv"

    df_aggr_part1 = pd.read_csv(aggr_part1_f)
    df_aggr_part2 = pd.read_csv(aggr_part2_f)

    output_filename = "coordinates_PILOT.js"
    output_file = open(output_filename,"w")

    dat = {}
    names = []

    for idx, row in df_loc.iterrows():
        rec = {}
        loc_id = int(row["loc_id"])
        rec["name"] = loc_id
        rec["lat"] = row["latitude"]
        rec["long"] = row["longitude"]

        status = 0
        rec["overall_attractiveness"] = 0
        if(df_aggr_part2[df_aggr_part2["loc_id"]==loc_id].shape[0]>0):
            status = 1
            rec["overall_attractiveness"] = df_aggr_part2[df_aggr_part2["loc_id"]==loc_id]["median"].values[0]

        rec["status"] = status

        imgs = {}
        for i in range(1,5):
            im = {}
            ky = "img"+str(i)
            img_id = loc_im[loc_id][ky]
            filt = df_img[df_img["id"] == img_id]
            im["filepath"] = filt["filepath"].values[0]
            im["heading"] = filt["heading"].values[0]
            im["attractiveness"] = 0
            if status:
                im["attractiveness"] = df_aggr_part1[df_aggr_part1["img_id"]==img_id]["median"].values[0]
            imgs[ky] = im

        rec["imgs"] = imgs

        dat[rec["name"]] = rec
        names.append(rec["name"])

    #dat_json = json.JSONEncoder().encode(dat)
    output_file.write("var coordinates_data = "+str(dat))
    output_file.close()

def import_city_attractiveness(input_file="../../../../DATA/amsterdam_attractiveness.csv", output_name = "../Amsterdam/Data/att_amsterdam.js"):
    output_file = open(output_name, "w")

    dat = {}
    df_loc = pd.read_csv(input_file,",")

    loc_i = 0
    for idx, row in df_loc.iterrows():
        rec = {}
        loc_i = loc_i + 1
        loc_id = "loc_"+str(loc_i)
        rec["name"] = loc_id
        rec["lat"] = round(row["lat"],3)
        rec["long"] = round(row["long"],3)
        rec["overall_attractiveness"] = row["attractiveness"]
        dat[rec["name"]] = rec

    # dat_json = json.JSONEncoder().encode(dat)
    output_file.write("var coordinates_data = " + str(dat))
    output_file.close()
