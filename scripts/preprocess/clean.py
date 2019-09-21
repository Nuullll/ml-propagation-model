import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train_set")
TEST_DIR = os.path.join(DATASET_DIR, "test_set")
VISUAL_DIR = os.path.join(PROJECT_DIR, "visualization")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed_data")
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
PATH_DIR = os.path.join(PROJECT_DIR, 'path_data')
PATH_TRAIN_DIR = os.path.join(PATH_DIR, 'train')

TRAIN_CONFIG = {
    'xmin': 382930,
    'xmax': 434580,
    'ymin': 3375740,
    'ymax': 3418880,
    'tileSize': 2000,
    'step': 5,

    'trainset': 4000,
    'points': 12011833,
    'buckets': [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
}

CLUTTER_HEIGHT = {
    1: 0, 2: 0, 3: 0,
    4: 3, 5: 5, 6: 1,
    7: 1, 8: 1, 9: 3,
    10: 80, 11: 50, 12: 30,
    13: 20, 14: 10, 15: 10,
    16: 30, 17: 5, 18: 10,
    19: 4, 20: 80
}


class Antenna:
    def __init__(self, x, y, azimuth, eDowntilt, mDowntilt, freq, power):
        self.x = x
        self.y = y
        self.azimuth = azimuth      # deg
        self.eDowntilt = eDowntilt  # deg
        self.mDowntilt = mDowntilt  # deg
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

        xmin = max(xmin, TRAIN_CONFIG['xmin'])
        xmax = min(xmax, TRAIN_CONFIG['xmax'])
        ymin = max(ymin, TRAIN_CONFIG['ymin'])
        ymax = min(ymax, TRAIN_CONFIG['ymax'])

        if abs(self.azimuth) < 20 or abs(self.azimuth-180) < 20:
            y = np.linspace(ymin, ymax, N)
            x = (y - self.y) * math.tan(rad) + self.x
        else:
            x = np.linspace(xmin, xmax, N)
            y = (x - self.x) / math.tan(rad) + self.y

        plt.plot(x, y, color='#0000ff', linestyle='dashed', linewidth=0.3, alpha=0.5)
        plt.axis([TRAIN_CONFIG['xmin'], TRAIN_CONFIG['xmax'], TRAIN_CONFIG['ymin'], TRAIN_CONFIG['ymax']])


class Station:
    def __init__(self, cellIndex, x, y, height, azimuth, eDowntilt, mDowntilt, freq, power,
                 buildingHeight, altitude, clutter):
        self.cellIndex = int(cellIndex)
        self.x = x
        self.y = y
        self.height = height        # m
        self.buildingHeight = buildingHeight    # m
        self.altitude = altitude                # m
        self.clutter = clutter

        self.antenna = Antenna(x=x, y=y, azimuth=azimuth,
                               eDowntilt=eDowntilt, mDowntilt=mDowntilt, freq=freq, power=power)

    def plot(self, xrange, yrange):
        self.antenna.plot(xrange, yrange)


def loadMap(filename):
    df = pd.read_csv(filename)

    # assert ok on train_set: 4000/4000
    # gk = df.groupby(['Cell X', 'Cell Y'])
    # assert len(gk) == 1

    # get station info
    first = df.loc[0, :]
    station = Station(cellIndex=first['Cell Index'], x=first['Cell X'], y=first['Cell Y'],
                      height=first['Height'], azimuth=first['Azimuth'], eDowntilt=first['Electrical Downtilt'],
                      mDowntilt=first['Mechanical Downtilt'], freq=first['Frequency Band'], power=first['RS Power'],
                      buildingHeight=first['Cell Building Height'], altitude=first['Cell Altitude'],
                      clutter=first['Cell Clutter Index'])

    return df, station


def walkDataset(dataset_dir):

    for root, _, files in os.walk(dataset_dir):

        for file in files:
            fullname = os.path.join(root, file)
            _, ext = os.path.splitext(fullname)

            if ext == '.csv':
                yield(fullname)


def aggregate(dir):

    count = 0
    total = TRAIN_CONFIG['trainset']

    for csv in walkDataset(dir):
        df, station = loadMap(csv)

        df['Station Absolute Height'] = df["Height"] + df["Cell Altitude"]
        df["X Distance"] = df["X"] - df["Cell X"]
        df["Y Distance"] = df["Y"] - df["Cell Y"]
        df["Distance To Station"] = (df["X Distance"].pow(2) + df["Y Distance"].pow(2)).pow(1./2)
        df["Altitude Delta"] = df["Altitude"] - df["Cell Altitude"]
        df["Azimuth To Station"] = np.degrees(np.arctan2(df["Y Distance"], df["X Distance"])) + 180 - df["Azimuth"]
        df["Height Delta"] = df["Altitude"] - df["Station Absolute Height"]
        df["Station Total Downtilt"] = df["Electrical Downtilt"] + df["Mechanical Downtilt"]
        df["Station Downtilt Delta"] = df["Electrical Downtilt"] - df["Mechanical Downtilt"]
        df["Vertical Degree"] = np.degrees(np.arctan2(-df["Height Delta"], df["Distance To Station"]))

        df.to_csv(os.path.join(PROCESSED_DIR, f"cell_agg_{station.cellIndex}.csv"))

        count += 1
        print(f"Task ok: {count}/{total}")


def aggregatePath():

    count = 0
    total = TRAIN_CONFIG['trainset']
    step = TRAIN_CONFIG['step']

    for csv in walkDataset(PROCESSED_TRAIN_DIR):
        df, station = loadMap(csv)

        d = df[["X", "Y"]].to_dict('records')
        lookup = {(m["X"], m["Y"]): i for i, m in enumerate(d)}
        marked = set()

        xmin = df["X"].min()
        xmax = df["X"].max()
        ymin = df["Y"].min()
        ymax = df["Y"].max()

        xstep = step
        ystep = step
        for a in range(0, 360, 3):
            ra = math.radians(a)

            if a <= 90:
                xend = xmax
                yend = ymax
            elif a <= 180:
                xend = xmax
                yend = ymin
                ystep = -step
            elif a <= 270:
                xend = xmin
                yend = ymin
                xstep = -step
                ystep = -step
            else:
                xend = xmin
                yend = ymax
                xstep = -step

            if a % 180 == 0:
                ylist = np.arange(station.y, yend+ystep, ystep)
                xlist = [station.x for _ in ylist]
            elif a % 90 == 0:
                xlist = np.arange(station.x, xend+xstep, xstep)
                ylist = [station.y for _ in xlist]
            else:
                xlist1 = np.arange(station.x, xend+xstep, xstep)
                ylist1 = np.floor(((xlist1-station.x) / math.tan(ra) + station.y) / 5) * 5
                ylist2 = np.arange(station.y, yend+ystep, ystep)
                xlist2 = np.floor((ylist2-station.y) * math.tan(ra) + station.x / 5) * 5

                xlist = list(xlist1) + list(xlist2)
                ylist = list(ylist1) + list(ylist2)

                zip_list = list(set(zip(xlist, ylist)))
                zip_list.sort(key=lambda x: (x[0]-station.x)**2 + (x[1]-station.y)**2)

                xlist, ylist = zip(*zip_list)

            ref_alt = station.altitude
            station_h = station.height
            waypoints_h = []
            waypoints_d = []        # dis from waypoint to station
            for x, y in zip(xlist, ylist):
                if (x, y) not in lookup:
                    continue

                i = lookup[(x, y)]
                row = df.loc[i, :]
                vdeg = row["Vertical Degree"]
                alt = row["Altitude"]
                ref_h = station_h + ref_alt - alt
                waypoints_h.append(max(CLUTTER_HEIGHT[int(row["Clutter Index"])], row["Building Height"]) + alt)
                waypoints_d.append(distance(x, y, station.x, station.y))

                if (x, y) in marked:
                    continue

                bins = binShot(waypoints_h, waypoints_d, ref_h, vdeg, alt)
                for k, v in bins.items():
                    df.loc[i, f"bin{k}"] = v

                marked.add((x, y))

            print(f"Scanning degree: {a} | {count}/{total}")

        df.to_csv(os.path.join(PATH_TRAIN_DIR, f"cell_path_agg_{station.cellIndex}.csv"))
        count += 1


def binShot(hs, ds, ref_h, vdeg, alt):
    buckets = TRAIN_CONFIG['buckets']
    binSize = buckets[1]-buckets[0]

    bins = {key: 0 for key in buckets}
    for h, d in zip(hs, ds):
        e = h-alt+d*math.tan(math.radians(vdeg))
        b = int(max(0, min(e/ref_h/binSize, len(buckets)-1)))
        bins[buckets[b]] += 1

    return bins


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


if __name__ == "__main__":
    aggregatePath()
