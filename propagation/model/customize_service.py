import os
import sys
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PACKAGE_DIR)

from model_service.tfserving_model_service import TfServingBaseService
from preprocess import Aggregator
from sklearn.externals import joblib
import os
from lgbmodel import LGBModel


class CustomService(TfServingBaseService):

    def _preprocess(self, data):
        if not hasattr(self, "myLGBModel"):
            self.myMode = "BASELINE"
            model_file = os.path.join(self.model_path, "lgb.pkl")
            self.myLGBModel = LGBModel(joblib.load(model_file))
            print("Model loaded.")

        if self.myMode == "BASELINE":
            return self._baseline_preprocess(data)

        elif self.myMode == "AGGREGATE":
            return self._aggregate_preprocess(data)

        return data

    def _inference(self, data):
        # override default inference method
        if self.myMode == "BASELINE":
            file = data['original_file']
        elif self.myMode == "AGGREGATE":
            file = data['aggregated_file']
        else:
            file = None

        predict_data = {}

        result, cell_id = self.myLGBModel.predict(file)
    
        predict_data["RSRP"] = result
        predict_data["cell_index"] = cell_id

        return predict_data

    def _postprocess(self, data):
        res_list = []

        with open("test_{}.csv_result.txt".format(data["cell_index"]), "w") as f:
            for val in data["RSRP"]:
                res_list.append([val])
                f.write("{}\n".format(val))

        return {"RSRP": res_list}

    def _baseline_preprocess(self, data):
        # preprocess here
        preprocessed_data = {}
        # files = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                if not file_name.endswith(".csv"):
                    continue

                preprocessed_data['original_file'] = file_content

        return preprocessed_data

    def _aggregate_preprocess(self, data):
        # preprocess here
        preprocessed_data = {}
        # files = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                if not file_name.endswith(".csv"):
                    continue

                worker = Aggregator(filebuffer=file_content, identifier=file_name)
                agg_filename = worker.run()

                preprocessed_data['aggregated_file'] = agg_filename

        return preprocessed_data
