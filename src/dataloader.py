import json

import torch
import numpy as np
import xlrd


class DataLoader:
    def __init__(self):
        self.device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_parameters_data_file = "../data/model_parameters.json"
        self._contact_data_file = "../data/contact_matrices.xls"
        self._age_data_file = "../data/age_distribution.xls"
        self._model_structure_file = "../model_struct.json"

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()
        self.param_names = np.array([f'daily_vac_{i}' for i in range(self.contact_data["home"].shape[0])])

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = torch.Tensor([sheet.row_values(i) for i in range(0, sheet.nrows)])
        wb.unload_sheet(0)
        self.age_data = datalist.to(self.device)

    def _get_model_parameters_data(self):
        # Load model param_names
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data.update({param: torch.Tensor(param_value).to(self.device)})
            else:
                self.model_parameters_data.update({param: param_value})

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        for idx in range(4):
            sheet = wb.sheet_by_index(idx)
            datalist = torch.Tensor([sheet.row_values(i) for i in range(0, sheet.nrows)])
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            datalist = self.transform_matrix(datalist.to(self.device))
            contact_matrices.update({cm_type: datalist})
        self.contact_data = contact_matrices

    def transform_matrix(self, matrix: torch.Tensor):
        """
        Perform symmetrization of contact matrix, by averaging it's elements with the elements of
        it's transpose, then dividing each row with the size of the age group corresponding to the given row.

        After this, the C[i, j] element represents the average number of interactions a member of age
        group i has with members of age group j, while C[j, i] represents the average number of
        interactions a member of age group j has with members of age group j in a day.

        For example, in the case that the population of age group i is the double of age group j:

                                     C[j, i] = 2 * C[i, j]

        Args:
            matrix (torch.Tensor): The contact matrix.

        Returns:
            torch.Tensor: The transformed contact matrix.
        """
        # Get age vector as a column vector
        age_distribution = self.age_data.reshape((-1, 1))
        # Get matrix of total number of contacts
        matrix_1 = matrix * age_distribution
        # Get symmetrized matrix
        output = (matrix_1 + matrix_1.T) / 2
        # Get contact matrix
        output /= age_distribution
        return output

    def _load_model_structure(self):
        with open(self._model_structure_file) as f:
            model_structure = json.load(f)

        self.state_data = model_structure["states"]
        self.transition_data = model_structure["transitions"]
