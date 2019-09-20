import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pymongo
from pymongo.collection import ReturnDocument


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train_set")
VISUAL_DIR = os.path.join(PROJECT_DIR, "visualization")

TRAIN_CONFIG = {
    'xmin': 382930,
    'xmax': 434580,
    'ymin': 3375740,
    'ymax': 3418880,
    'tileSize': 2000,
    'step': 5,

    'trainset': 4000,
}

client = pymongo.MongoClient()

db = client.propagtion_map
stations = db.stations
points = db.points
antennas = db.antennas

stations.create_index([("x", pymongo.ASCENDING), ("y", pymongo.ASCENDING)], unique=True)
points.create_index([("x", pymongo.ASCENDING), ("y", pymongo.ASCENDING)], unique=True)
antennas.create_index([("x", pymongo.ASCENDING), ("y", pymongo.ASCENDING), ("azimuth", pymongo.ASCENDING)], unique=True)


# class MapTile:
#     def __init__(self, xmin, ymin):
#         size = TRAIN_CONFIG['tileSize']
#         step = TRAIN_CONFIG['step']
#
#         self.xmin = xmin
#         self.xmax = xmin + size - step
#         self.ymin = ymin
#         self.ymax = ymin + size - step
#         self.tileSize = size
#         self.step = step
#
#         self.details = {}       # (x,y) as key
#         self.stations = {}      # (x,y) as key
#
#     @classmethod
#     def load(cls, x, y):
#         filename, xmin, ymin = cls.locateFile(x, y)
#         if os.path.exists(filename):
#             with open(filename, 'r') as f:
#                 return jsonpickle.decode(f.read())
#         else:
#             return MapTile(xmin, ymin)
#
#     @classmethod
#     def locateTile(cls, x, y):
#         size = TRAIN_CONFIG['tileSize']
#         xmin = (x // size) * size
#         ymin = (y // size) * size
#
#         return xmin, ymin
#
#     @classmethod
#     def locateFile(cls, x, y):
#         size = TRAIN_CONFIG['tileSize']
#         xmin, ymin = cls.locateTile(x, y)
#         return os.path.join(VISUAL_DIR, 'metamap', f"tile{size}_{xmin}_{ymin}.json"), xmin, ymin
#
#     def save(self):
#         filename, _, _ = self.locateFile(self.xmin, self.ymin)
#
#         with open(filename, 'w') as f:
#             f.write(jsonpickle.encode(self))
#
#     def addStation(self, station):
#         if (station.x, station.y) not in self.stations:
#             self.stations[(station.x, station.y)] = station
#
#     def addPoint(self, point):
#         if (point.x, point.y) not in self.details:
#             self.details[(point.x, point.y)] = point


class Point:
    def __init__(self, x, y, altitude, buildingHeight, clutter, RSRP):
        self.x = x
        self.y = y
        self.altitude = altitude
        self.buildingHeight = buildingHeight
        self.clutter = clutter
        self.RSRP = RSRP


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


def heatMap(data, station):
    """
    Plot head map of RSRP
    :param data:    <pd.DataFrame>
    :return:
    """
    signalMap = data[['X', 'Y', 'RSRP']]

    # normalize RSRP
    # rsrp = signalMap['RSRP']
    # norm_rsrp = (rsrp - rsrp.min()) / (rsrp.max() - rsrp.min())

    plt.scatter(signalMap['X'], signalMap['Y'], s=0.3, c=signalMap['RSRP'], cmap='Reds')
    # cb = plt.colorbar()
    # cb.set_label('Normalized RSRP', rotation=270, labelpad=20)

    # plot station
    plt.scatter(station.x, station.y, s=2, c='#0000ff')
    xrange = (signalMap['X'].min(), signalMap['X'].max())
    yrange = (signalMap['Y'].min(), signalMap['Y'].max())
    station.plot(xrange, yrange)

    # plt.title(f"RSRP Heat Map of Cell {station.cellIndex}")

    # plt.savefig(os.path.join(VISUAL_DIR, 'heatmap', f"cell_{station.cellIndex}.jpg"))
    # plt.clf()


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


# def getMetaMap():
#     count = 1
#     for csv in walkDataset(TRAIN_DIR):
#         df, station = loadMap(csv)
#
#         # locate tile
#         tile = MapTile.load(station.x, station.y)
#         tile.addStation(station)
#
#         xmin = tile.xmin
#         ymin = tile.ymin
#         details = df.loc[:, ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index', 'RSRP']]
#
#         rowCount = 0
#         rowTotal = len(details)
#         for _, row in details.sort_values(['X', 'Y']).iterrows():
#             p = Point(row['X'], row['Y'], row['Altitude'], row['Building Height'], row['Clutter Index'], row['RSRP'])
#             xm, ym = MapTile.locateTile(p.x, p.y)
#
#             if not (xm == xmin and ym == ymin):
#                 # read new tile
#                 tile.save()
#                 tile = MapTile.load(p.x, p.y)
#                 xmin = tile.xmin
#                 ymin = tile.ymin
#
#             tile.addPoint(p)
#
#             rowCount += 1
#             print(f"Processing: row {rowCount}/{rowTotal}, cell {count}/{TRAIN_CONFIG['trainset']}")
#
#         tile.save()
#
#         count += 1


def saveStations():
    count = 0
    for csv in walkDataset(TRAIN_DIR):
        df, station = loadMap(csv)

        a = antennas.find_one_and_update(
            {
                "x": station.x,
                "y": station.y,
                "azimuth": station.antenna.azimuth
            },
            {
                "$set": {
                    "eDowntilt": station.antenna.eDowntilt,
                    "mDowntilt": station.antenna.mDowntilt,
                    "freq": station.antenna.freq,
                    "power": station.antenna.power,
                }
            }, upsert=True, return_document=ReturnDocument.AFTER
        )

        stations.find_one_and_update(
            {
                "x": station.x,
                "y": station.y,
            },
            {
                "$set": {
                    "cellIndex": station.cellIndex,
                    "height": station.height,
                    "buildingHeight": station.buildingHeight,
                    "altitude": station.altitude,
                    "clutterIndex": station.clutter,
                },
                "$push": {
                    "antennas": a["_id"]
                }
            }, upsert=True, return_document=ReturnDocument.AFTER
        )

        count += 1
        print(f"Stations saved: {count}/{TRAIN_CONFIG['trainset']}")


def savePoints():
    cellCount = 0
    cellTotal = TRAIN_CONFIG['trainset']

    for csv in walkDataset(TRAIN_DIR):
        df, station = loadMap(csv)
        details = df.loc[:, ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index', 'RSRP']]

        rowCount = 0
        rowTotal = len(details)
        for _, row in details.sort_values(['X', 'Y']).iterrows():
            p = Point(row['X'], row['Y'], row['Altitude'], row['Building Height'], row['Clutter Index'], row['RSRP'])

            points.find_one_and_update(
                {
                    "x": p.x,
                    "y": p.y,
                },
                {
                    "$set": {
                        "altitude": p.altitude,
                        "buildingHeight": p.buildingHeight,
                        "clutterIndex": p.clutter,
                        "RSRP": p.RSRP,
                    }
                }, upsert=True, return_document=ReturnDocument.AFTER
            )

            rowCount += 1
            print(f"Points saved: {rowCount}/{rowTotal} | {cellCount}/{cellTotal}")

        cellCount += 1


def hugeMap():
    cellCount = 0
    cellTotal = TRAIN_CONFIG['trainset']

    for csv in walkDataset(TRAIN_DIR):
        df, station = loadMap(csv)
        heatMap(df, station)

        cellCount += 1
        print(f"Heatmap integrated: {cellCount}/{cellTotal}")

        if cellCount % 500 == 1:
            plt.pause(0.1)


def aggregateStations():
    stations.aggregate([
        {
            "$addFields": {
                "absoluteAltitude": {
                    "$add": ["$altitude", "$height"]
                }
            }
        },
        {
            "$out": "stations"
        }
    ])


def aggregateAntennas():
    antennas.aggregate([
        {
            "$addFields": {
                "downtilt": {
                    "$add": ["$eDowntilt", "$mDowntilt"]
                },
                "downtiltDelta": {
                    "$subtract": ["$eDowntilt", "$mDowntilt"]
                }
            }
        },
        {
            "$out": "antennas"
        }
    ])


if __name__ == "__main__":
    aggregateAntennas()




