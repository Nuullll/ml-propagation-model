import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train_set")
VISUAL_DIR = os.path.join(PROJECT_DIR, "visualization")


class Station:
    def __init__(self, cellIndex, x, y, height, azimuth, eDowntilt, mDowntilt, freq, power):
        self.cellIndex = int(cellIndex)
        self.x = x
        self.y = y
        self.height = height        # m
        self.azimuth = azimuth      # deg
        self.eDowntilt = eDowntilt  # Deg
        self.mDowntilt = mDowntilt  # Deg
        self.freq = freq            # MHz
        self.power = power          # dBm

    def plot(self, xrange, yrange):
        N = 100
        rad = math.radians(self.azimuth)

        xmin, xmax = xrange
        ymin, ymax = yrange

        xsign = True if math.sin(rad) >= 0 else False
        ysign = True if math.cos(rad) >= 0 else False

        if xsign:
            xmin = max(xmin, self.x)
        else:
            xmax = min(xmax, self.x)

        if ysign:
            ymin = max(ymin, self.y)
        else:
            ymax = min(ymax, self.y)

        if self.azimuth == 0 or self.azimuth == 180:
            y = np.linspace(ymin, ymax, N)
            x = self.x * np.ones(N)
        else:
            x = np.linspace(xmin, xmax, N)
            y = (x - self.x) / math.tan(rad) + self.y

        plt.plot(x, y, color='#0000ff', linewidth=3, linestyle='dashed')


def heatMap(data, station):
    """
    Plot head map of RSRP
    :param data:    <pd.DataFrame>
    :return:
    """
    signalMap = data[['X', 'Y', 'RSRP']]

    # normalize RSRP
    rsrp = signalMap['RSRP']
    norm_rsrp = (rsrp - rsrp.min()) / (rsrp.max() - rsrp.min())

    plt.scatter(signalMap['X'], signalMap['Y'], c=norm_rsrp, cmap='Reds')
    cb = plt.colorbar()
    cb.set_label('Normalized RSRP', rotation=270, labelpad=20)

    # plot station
    plt.scatter(station.x, station.y, s=30, c='#0000ff')
    xrange = (signalMap['X'].min(), signalMap['X'].max())
    yrange = (signalMap['Y'].min(), signalMap['Y'].max())
    station.plot(xrange, yrange)

    plt.title(f"RSRP Heat Map of Cell {station.cellIndex}")

    plt.savefig(os.path.join(VISUAL_DIR, 'heatmap', f"cell_{station.cellIndex}.jpg"))
    plt.clf()


def loadMap(filename):
    df = pd.read_csv(filename)

    # assert ok on train_set: 4000/4000
    # gk = df.groupby(['Cell X', 'Cell Y'])
    # assert len(gk) == 1

    # get station info
    first = df.loc[0, :]
    station = Station(cellIndex=first['Cell Index'], x=first['Cell X'], y=first['Cell Y'],
                      height=first['Height'], azimuth=first['Azimuth'], eDowntilt=first['Electrical Downtilt'],
                      mDowntilt=first['Mechanical Downtilt'], freq=first['Frequency Band'], power=first['RS Power'])

    return df, station


def walkDataset(dataset_dir):

    for root, _, files in os.walk(dataset_dir):
        count = 0
        total = len(files)

        for file in files:
            fullname = os.path.join(root, file)
            _, ext = os.path.splitext(fullname)

            if ext == '.csv':
                df, station = loadMap(fullname)
                heatMap(df, station)

                count += 1
                print(f"Task ok: [{count}/{total}]")

        break


if __name__ == "__main__":
    walkDataset(TRAIN_DIR)
