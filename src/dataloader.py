import json
import os

import numpy as np
import xlrd


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class DataLoader:
    def __init__(self):
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data", "model_parameters.json")
        self._contact_data_file = os.path.join(PROJECT_PATH, "../data", "contact_matrices.xls")
        self._age_data_file = os.path.join(PROJECT_PATH, "../data", "age_distribution.xls")

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()
        self.param_names = np.array(["alpha", "gamma", "beta_0", "daily_vaccines", "t_start", "rho", "psi"])

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
        wb.unload_sheet(0)
        self.age_data = datalist

    def _get_model_parameters_data(self):
        # Load model param_names
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data.update({param: np.array(param_value)})
            else:
                self.model_parameters_data.update({param: param_value})

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        for idx in range(4):
            sheet = wb.sheet_by_index(idx)
            datalist = np.array([sheet.row_values(i) for i in range(0, sheet.nrows)])
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            datalist = self.transform_matrix(datalist)
            contact_matrices.update({cm_type: datalist})
        self.contact_data = contact_matrices

    def transform_matrix(self, matrix: np.ndarray):
        # Get age vector as a column vector
        age_distribution = self.age_data.reshape((-1, 1))
        # Get matrix of total number of contacts
        matrix_1 = matrix * age_distribution
        # Get symmetrized matrix
        output = (matrix_1 + matrix_1.T) / 2
        # Get contact matrix
        output /= age_distribution
        return output
