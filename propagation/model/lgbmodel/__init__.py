# interfacing custom model here!

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils import shuffle


PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))


class LGBModel:
    def __init__(self, model):
        self.model = model

        enc = OneHotEncoder()
        a = [i for i in range(21)]
        enc.fit(np.array(a).reshape(-1, 1))
        self.encoder = enc
        print('one hot is ready!')

    def predict(self, test_file):
        test_data = pd.read_csv(test_file)
        cell_id = test_data.loc[0, 'Cell Index']
        nouse_feature = ['Cell Index', 'Cell X', 'Cell Y', 'X', 'Y', 'flag']
        continuous_feature = ['Height', 'Azimuth', 'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
                              'RS Power', 'Cell Altitude', 'Cell Building Height', 'Altitude', 'Building Height']
        # continuous_feature=[
        #    'RS Power','Station Absolute Height', 'Distance To Station', 'Altitude Delta',
        #    'Azimuth To Station', 'Height Delta', 'Station Total Downtilt', 'Station Downtilt Delta', 'Vertical Degree']
        continuous_feature = [
            'RS Power']  # ,'Station Absolute Height', 'Distance To Station', 'Altitude Delta', 'Azimuth To Station', 'Height Delta', 'Station Total Downtilt', 'Station Downtilt Delta', 'Vertical Degree']
        y_feature = ['RSRP']
        discrete_feature = ['Cell Clutter Index', 'Clutter Index']
        # print(data.head(2))

        # test_x, test_y = test_data[continuous_feature], test_data.pop('RSRP')
        test_x = test_data[continuous_feature]
        enc = self.encoder
        for feature in discrete_feature:
            test_a = enc.transform(test_data[feature].values.reshape(-1, 1))
            test_x = sparse.hstack((test_x, test_a))

        res = self.model.predict(test_x)
        # judget(res, test_y)
        print(min(res), max(res))
        print(res)
        return res, cell_id


def test_lgbmodel():
    model_file = os.path.join(PACKAGE_DIR, "lgb.pkl")
    model = LGBModel(joblib.load(model_file))

    test_file = os.path.join(PACKAGE_DIR, "test_112501.csv")

    res = model.predict(test_file)


if __name__ == '__main__':
    test_lgbmodel()
