import pysal
import pandas as pd
import numpy as np




def load_data(input_file = "../../DATA/attr_judgments.csv"):
    df_dat = pd.read_csv(input_file)
    lats = df_dat["latitude"].values
    longs = df_dat["longitude"].values

    lats.shape = (lats.shape[0],1)
    longs.shape = (longs.shape[0], 1)

    coord_data = np.hstack([lats, longs])

    y = np.array(df_dat["attractiveness"].values)

    return [coord_data, y]


def global_autocorrel():
    [coord_data, y] = load_data()
    wknn3 = pysal.weights.KNN(coord_data, k=3)

    wdist = pysal.weights.DistanceBand.from_array(coord_data, 0.03)

    w = wknn3
    w = wdist
    mi = pysal.Moran(y, w)
    mi.I

    for k in [1,2,3,4,5,6,7,8,9,10]:
        w = pysal.weights.KNN(coord_data, k=k)
        mi = pysal.Moran(y, w)
        print(str(k)+";"+str(mi.I)+";"+str(mi.EI)+";"+str(mi.p_norm))

    for th in [0.004, 0.003, 0.002, 0.001]:
        w = pysal.weights.DistanceBand.from_array(coord_data, th)
        mi = pysal.Moran(y, w)
        print(str(th)+";"+str(mi.I)+";"+str(mi.EI)+";"+str(mi.p_norm))

def local_autocorrel():
    [coord_data, y] = load_data()

    lm = pysal.Moran_Local(y, w)

main()