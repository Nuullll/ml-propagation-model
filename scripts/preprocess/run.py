import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pymongo
from pymongo.collection import ReturnDocument

R2D = 180 / math.pi

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
    'points': 12011833,
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

client = pymongo.MongoClient()

db = client.propagtion_map
stations = db.stations_combine
points = db.points
points_agg = db.points_agg
# antennas = db.antennas
points_full = db.points_full
points_small = db.points_small

# stations.create_index([("x", pymongo.ASCENDING), ("y", pymongo.ASCENDING), ("cellIndex", pymongo.ASCENDING),
#                        ("azimuth", pymongo.ASCENDING)], unique=True)
# points_full.create_index([("X", pymongo.ASCENDING), ("Y", pymongo.ASCENDING), ("Cell Index", pymongo.ASCENDING)], unique=True)
# antennas.create_index([("x", pymongo.ASCENDING), ("y", pymongo.ASCENDING), ("azimuth", pymongo.ASCENDING)], unique=True)


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


# class Point:
#     def __init__(self, x, y, altitude, buildingHeight, clutter, RSRP):
#         self.x = x
#         self.y = y
#         self.altitude = altitude
#         self.buildingHeight = buildingHeight
#         self.clutter = clutter
#         self.RSRP = RSRP


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


# def saveStations():
#     count = 0
#     for csv in walkDataset(TRAIN_DIR):
#         df, station = loadMap(csv)
#
#         a = antennas.find_one_and_update(
#             {
#                 "x": station.x,
#                 "y": station.y,
#                 "azimuth": station.antenna.azimuth
#             },
#             {
#                 "$set": {
#                     "eDowntilt": station.antenna.eDowntilt,
#                     "mDowntilt": station.antenna.mDowntilt,
#                     "freq": station.antenna.freq,
#                     "power": station.antenna.power,
#                 }
#             }, upsert=True, return_document=ReturnDocument.AFTER
#         )
#
#         stations.find_one_and_update(
#             {
#                 "x": station.x,
#                 "y": station.y,
#             },
#             {
#                 "$set": {
#                     "cellIndex": station.cellIndex,
#                     "height": station.height,
#                     "buildingHeight": station.buildingHeight,
#                     "altitude": station.altitude,
#                     "clutterIndex": station.clutter,
#                 },
#                 "$push": {
#                     "antennas": a["_id"]
#                 }
#             }, upsert=True, return_document=ReturnDocument.AFTER
#         )
#
#         count += 1
#         print(f"Stations saved: {count}/{TRAIN_CONFIG['trainset']}")


def saveStationsCombine():
    count = 0
    for csv in walkDataset(TRAIN_DIR):
        df, station = loadMap(csv)

        stations.find_one_and_update(
            {
                "x": station.x,
                "y": station.y,
                "cellIndex": station.cellIndex,
                "azimuth": station.antenna.azimuth,
            },
            {
                "$set": {
                    "height": station.height,
                    "buildingHeight": station.buildingHeight,
                    "altitude": station.altitude,
                    "clutterIndex": station.clutter,
                    "eDowntilt": station.antenna.eDowntilt,
                    "mDowntilt": station.antenna.mDowntilt,
                    "freq": station.antenna.freq,
                    "power": station.antenna.power,
                },
            }, upsert=True, return_document=ReturnDocument.AFTER
        )

        count += 1
        print(f"Stations saved: {count}/{TRAIN_CONFIG['trainset']}")


def savePoints():
    cellCount = 0
    cellTotal = TRAIN_CONFIG['trainset']

    for csv in walkDataset(TRAIN_DIR):
        df, station = loadMap(csv)
        details = df.loc[:, ['Cell Index', 'X', 'Y', 'Altitude', 'Building Height', 'Clutter Index', 'RSRP']]
        details = details.rename(columns={
            'Cell Index': 'cellIndex',
            'X': 'x', 'Y': 'y',
            'Altitude': 'altitude', 'Building Height': 'buildingHeight',
            'Clutter Index': 'clutterIndex'
        })

        p_list = details.to_dict('records')
        points.insert_many(p_list)

        cellCount += 1
        print(f"Points saved: {cellCount}/{cellTotal}")


def savePointsFull():
    cellCount = 0
    cellTotal = TRAIN_CONFIG['trainset']

    for csv in walkDataset(TRAIN_DIR):
        df, station = loadMap(csv)

        p_list = df.to_dict('records')
        points_full.insert_many(p_list, ordered=False)

        cellCount += 1
        print(f"Points saved: {cellCount}/{cellTotal}")


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


# def aggregateAntennas():
#     antennas.aggregate([
#         {
#             "$addFields": {
#                 "downtilt": {
#                     "$add": ["$eDowntilt", "$mDowntilt"]
#                 },
#                 "downtiltDelta": {
#                     "$subtract": ["$eDowntilt", "$mDowntilt"]
#                 }
#             }
#         },
#         {
#             "$out": "antennas"
#         }
#     ])


def aggregatePoints():
    points_agg_new = db.points_agg_new
    points_agg_new.aggregate([
        {
            "$addFields": {
                "Station Absolute Height": {
                    "$add": ["$Height", "$Cell Altitude"]
                },

                "X Distance": {
                    "$subtract": ["$X", "$Cell X"]
                },
                "Y Distance": {
                    "$subtract": ["$Y", "$Cell Y"]
                },

                "Distance To Station": {
                    "$sqrt": {
                        "$add": [
                            {
                                "$multiply": ["$X Distance", "$X Distance"]
                            },
                            {
                                "$multiply": ["$Y Distance", "$Y Distance"]
                            }
                        ]
                    }
                },

                "Altitude Delta": {
                    "$subtract": ["$Altitude", "$Cell Altitude"]
                },

                "Azimuth To Station": {
                    "$subtract": [
                        {
                            "$add": [180, {
                                "$radiansToDegrees": {"$atan2": ["$Y Distance", "$X Distance"]}
                            }]
                        },
                        "$Azimuth"
                    ]
                },

                "Height Delta": {
                    "$subtract": [
                        "$Altitude", "$Station Absolute Height"
                    ]
                },

                "Station Total Downtilt": {
                    "$add": [
                        "$Electrical Downtilt", "$Mechanical Downtilt"
                    ]
                },

                "Station Downtilt Delta": {
                    "$subtract": [
                        "$Electrical Downtilt", "$Mechanical Downtilt"
                    ]
                },

                "Vertical Degree": {
                    "$radiansToDegrees": {"$atan2": [{
                        "$subtract": [
                            0, "$Height Delta"
                        ]
                    }, "$Distance To Station"]}
                },
            }
        }, {
            "$out": "points_agg_new"
        }
    ])


def binIndex(hs, hx):
    i = int(hx/hs/0.2)
    return min(5, i)


def aggregatePath():
    total = TRAIN_CONFIG['points']
    count = 0
    step = TRAIN_CONFIG['step']

    for p in db.points_agg_new.find():
        xs = p['Cell X']
        ys = p['Cell Y']
        xd = p['X']
        yd = p['Y']

        xds = xd - xs
        yds = yd - ys

        hs = p['Station Absolute Height']
        alt = p['Altitude']
        a = p['Vertical Degree']
        ta = math.tan(math.radians(a))

        bins = [0]*6    # 0, 0.2, 0.4, 0.6, 0.8, 1.0(-1.2)
        bs = max(p['Cell Building Height'], CLUTTER_HEIGHT[int(p['Cell Clutter Index'])]) - p['Altitude Delta']
        i = binIndex(hs, bs)
        bins[i] += 1

        bd = max(p['Building Height'], CLUTTER_HEIGHT[int(p['Clutter Index'])])
        i = binIndex(hs, bd + p['Distance To Station'] * ta)
        bins[i] += 1

        visited = set()
        xstep = step if xd > xs else -step
        for x in range(int(xs)+xstep, int(xd), xstep):
            y = yds / xds * (x - xs) + ys
            y = int(y/5)*5

            pt = points.find_one({
                'X': x,
                'Y': y
            })
            if pt is None:
                continue

            b = max(pt['Building Height'], CLUTTER_HEIGHT[int(pt['Clutter Index'])]) + pt['Altitude'] - alt
            i = binIndex(hs, b + distance(pt['X'], pt['Y'], xs, ys) * ta)
            bins[i] += 1

            visited.add((x, y))

        ystep = step if yd > ys else -step
        for y in range(int(ys)+ystep, int(yd), ystep):
            x = xds / yds * (y - ys) + xs
            x = int(x / 5) * 5

            if (x, y) in visited:
                continue

            pt = points.find_one({
                'x': x,
                'y': y
            })
            if pt is None:
                continue

            b = max(pt['Building Height'], CLUTTER_HEIGHT[int(pt['Clutter Index'])]) + pt['Altitude'] - alt
            i = binIndex(hs, b + distance(pt['X'], pt['Y'], xs, ys) * ta)
            bins[i] += 1

        update_list = {f"bin{i}": v for i, v in enumerate(bins)}

        db.points_agg_new.update_one({
            "_id": p["_id"]
        }, {
            "$set": update_list
        })

        count += 1
        print(f"Path aggregated: {count}/{total}")


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def delta(alpha, beta):
    d = beta - alpha

    if d > 180:
        d -= 360
    elif d < -180:
        d += 360

    return d


# def findNearestStations(point, threshold=500):
#     x = point['x']
#     y = point['y']
#
#     docs = stations.find({
#         "x": {
#             "$gte": x - threshold,
#             "$lte": x + threshold
#         },
#         "y": {
#             "$gte": y - threshold,
#             "$lte": y + threshold,
#         }
#     })
#
#     if docs.count() < 3:
#         return findNearestStations(point, 2*threshold)
#
#     ants = []
#     for s in docs:
#         dis = distance(x, y, s['x'], s['y'])
#         deg2North = math.degrees(math.acos((x-s['x'])/dis))
#
#         minDelta = 180
#         best = None
#         for id in s["antennas"]:
#             a = antennas.find_one({"_id": id})
#             d = delta(a['azimuth'], deg2North)
#             if abs(d) <= abs(minDelta):
#                 minDelta = d
#                 best = a
#
#         ants.append((best, dis))
#
#     ants.sort(key=itemgetter(1))
#
#     ants = [a for a, _ in ants]
#     return ants[:3]


if __name__ == "__main__":
    aggregatePath()



