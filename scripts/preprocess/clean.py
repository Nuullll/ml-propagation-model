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
PATH_DIR = os.path.join(PROJECT_DIR, 'path_detail_data')
PATH_TRAIN_DIR = os.path.join(PATH_DIR, 'train')
FULL_PATH_DIR = os.path.join(PROJECT_DIR, "full_path_data")
FULL_PATH_TRAIN_DIR = os.path.join(FULL_PATH_DIR, "train")
COMPRESS_DIR = os.path.join(PROJECT_DIR, "compress_data")
COMPRESS_TRAIN_DIR = os.path.join(COMPRESS_DIR, "train")

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


def mkdirs(dir):
    try:
        os.makedirs(dir)
    except FileExistsError as _:
        pass


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

        files.sort()

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
        df["Azimuth To Station"] = (np.degrees(np.arctan2(df["X Distance"], df["Y Distance"])) + 360) % 360 - df["Azimuth"]
        df["Height Delta"] = df["Altitude"] - df["Station Absolute Height"]
        df["Station Total Downtilt"] = df["Electrical Downtilt"] + df["Mechanical Downtilt"]
        df["Station Downtilt Delta"] = df["Electrical Downtilt"] - df["Mechanical Downtilt"]
        df["Vertical Degree"] = np.degrees(np.arctan2(-df["Height Delta"], df["Distance To Station"]))
        df["Vertical Degree Delta"] = df["Vertical Degree"] - df["Station Total Downtilt"]

        df.to_csv(os.path.join(PROCESSED_DIR, f"cell_agg_{station.cellIndex}.csv"))

        count += 1
        print(f"Task ok: {count}/{total}")


def getPath(a, xmin, xmax, ymin, ymax, xs, ys):
    step = TRAIN_CONFIG['step']

    ra = math.radians(a)

    xstep = step
    ystep = step
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
        ylist = np.arange(ys, yend + ystep, ystep)
        xlist = [xs for _ in ylist]
    elif a % 90 == 0:
        xlist = np.arange(xs, xend + xstep, xstep)
        ylist = [ys for _ in xlist]
    else:
        xlist1 = np.arange(xs, xend + xstep, xstep)
        ylist1 = np.floor(((xlist1 - xs) / math.tan(ra) + ys) / 5) * 5
        ylist2 = np.arange(ys, yend + ystep, ystep)
        xlist2 = np.floor(((ylist2 - ys) * math.tan(ra) + xs) / 5) * 5

        xlist = list(xlist1) + list(xlist2)
        ylist = list(ylist1) + list(ylist2)

        zip_list = list(set(zip(xlist, ylist)))
        zip_list.sort(key=lambda x: (x[0] - xs) ** 2 + (x[1] - ys) ** 2)

        if len(zip_list) == 0:
            return [], []

        xlist, ylist = zip(*zip_list)

    return xlist, ylist


def aggregatePath():

    count = 0
    total = TRAIN_CONFIG['trainset']
    mkdirs(PATH_TRAIN_DIR)

    for csv in walkDataset(PROCESSED_TRAIN_DIR):
        df, station = loadMap(csv)

        target_file = os.path.join(PATH_TRAIN_DIR, f"cell_path_agg_{station.cellIndex}.csv")
        if os.path.exists(target_file):
            count += 1
            continue

        d = df[["X", "Y"]].to_dict('records')
        lookup = {(m["X"], m["Y"]): i for i, m in enumerate(d)}
        marked = set()

        xmin = df["X"].min()
        xmax = df["X"].max()
        ymin = df["Y"].min()
        ymax = df["Y"].max()

        for a in range(0, 360, 3):
            xlist, ylist = getPath(a, xmin, xmax, ymin, ymax, station.x, station.y)

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

                df.loc[i, "path_h"] = " ".join(map(str, map(int, waypoints_h)))
                df.loc[i, "path_d"] = " ".join(map(str, map(int, waypoints_d)))

                marked.add((x, y))

            print(f"Scanning degree: {a} | {count}/{total}")

        df.to_csv(target_file)
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


def fillMissingValue(dir):

    count = 0
    total = TRAIN_CONFIG['trainset']
    mkdirs(FULL_PATH_TRAIN_DIR)

    for csv in walkDataset(dir):
        df, station = loadMap(csv)

        target_file = os.path.join(FULL_PATH_TRAIN_DIR, f"cell_full_path_{station.cellIndex}.csv")
        if os.path.exists(target_file):
            count += 1
            continue

        mask = df["bin0"].isnull()
        vac = mask.sum()
        vcount = 0

        d = df[["X", "Y"]].to_dict('records')
        lookup = {(m["X"], m["Y"]): i for i, m in enumerate(d)}
        marked = set(k for k, i in lookup.items() if not mask[i])

        wait_list = df[df["bin0"].isna()]

        for _, row in wait_list.iterrows():
            x = row["X"]
            y = row["Y"]
            if (x, y) in marked:
                continue

            a = math.degrees(math.atan2(x - station.x, y - station.y) + 2*math.pi)
            a = a - 360 if a >= 360 else a

            xlist, ylist = getPath(a, x, x, y, y, station.x, station.y)

            ref_alt = station.altitude
            station_h = station.height
            waypoints_h = []
            waypoints_d = []
            prevx, prevy = None, None
            for x, y in zip(xlist, ylist):
                if (x, y) not in lookup:
                    continue

                i = lookup[(x, y)]
                r = df.loc[i, :]

                if (x, y) in marked:
                    prevx, prevy = x, y
                    continue

                if prevx is not None:
                    waypoints_h = list(map(int, df.loc[lookup[(prevx, prevy)], "path_h"].split()))
                    waypoints_d = list(map(int, df.loc[lookup[(prevx, prevy)], "path_d"].split()))

                vdeg = r["Vertical Degree"]
                alt = r["Altitude"]
                ref_h = station_h + ref_alt - alt
                waypoints_h.append(max(CLUTTER_HEIGHT[int(r["Clutter Index"])], r["Building Height"]) + alt)
                waypoints_d.append(distance(x, y, station.x, station.y))

                bins = binShot(waypoints_h, waypoints_d, ref_h, vdeg, alt)
                for k, v in bins.items():
                    df.loc[i, f"bin{k}"] = v

                df.loc[i, "path_h"] = " ".join(map(str, map(int, waypoints_h)))
                df.loc[i, "path_d"] = " ".join(map(str, map(int, waypoints_d)))

                marked.add((x, y))

                vcount += 1
                print(f"Vacancy filled: {vcount}/{vac} | {count}/{total}")

        df.to_csv(target_file)
        count += 1


def clean(dir):
    count = 0
    total = TRAIN_CONFIG['trainset']
    mkdirs(COMPRESS_TRAIN_DIR)

    for csv in walkDataset(dir):
        df, station = loadMap(csv)

        target_file = os.path.join(COMPRESS_TRAIN_DIR, f"compress_{station.cellIndex}.csv")
        if os.path.exists(target_file):
            count += 1
            continue

        df["Vertical Degree Delta"] = df["Vertical Degree"] - df["Station Total Downtilt"]

        cols = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'X Distance', 'Y Distance', 'path_h', 'path_d']
        try:
            df = df.drop(cols, axis=1)
        except Exception as e:
            pass

        df.to_csv(target_file)

        count += 1
        print(f"OK: {count}/{total}")


if __name__ == "__main__":
    clean(FULL_PATH_TRAIN_DIR)
