
import os
import pandas as pd
import numpy as np
import math


CONFIG = {
    'step': 5,
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

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
TMP_DIR = os.path.join(PACKAGE_DIR, "tmp")


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


def loadMap(filebuffer):
    df = pd.read_csv(filebuffer)

    # get station info
    first = df.loc[0, :]
    station = Station(cellIndex=first['Cell Index'], x=first['Cell X'], y=first['Cell Y'],
                      height=first['Height'], azimuth=first['Azimuth'], eDowntilt=first['Electrical Downtilt'],
                      mDowntilt=first['Mechanical Downtilt'], freq=first['Frequency Band'], power=first['RS Power'],
                      buildingHeight=first['Cell Building Height'], altitude=first['Cell Altitude'],
                      clutter=first['Cell Clutter Index'])

    return df, station


def getPath(a, xmin, xmax, ymin, ymax, xs, ys):
    step = CONFIG['step']

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


class Aggregator:
    def __init__(self, filebuffer, identifier="default"):
        self.name = identifier
        self.df, self.station = loadMap(filebuffer)
        self.id = self.station.cellIndex

    def log(self, message):
        print("[{}]: {}".format(self.name, message))

    def addFields(self):
        self.log("Adding fields...")

        df = self.df

        df["Station Absolute Height"] = df["Height"] + df["Cell Altitude"]
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


    def scanPath(self):
        self.log("Scanning paths on 360deg directions...")

        df, station = self.df, self.station

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

    def generateAllPath(self):
        self.log("Generating missing paths...")

        df, station = self.df, self.station

        mask = df["bin0"].isnull()

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

    def export(self):
        target_dir = os.path.join(TMP_DIR, str(self.id))
        mkdirs(target_dir)

        target_file = os.path.join(target_dir, "test_{}_agg.csv".format(self.id))

        self.log("Writing to {}".format(target_file))
        self.df.to_csv(target_file)

        return target_file

    def run(self):
        self.log("Start aggregating...")
        
        self.addFields()
        self.scanPath()
        self.generateAllPath()

        return self.export()