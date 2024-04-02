from abc import ABC
from os.path import dirname, realpath

PROJECT_PATH = dirname(dirname(realpath(__file__))).replace('\\', "/")


class DataLoaderBase(ABC):
    def __init__(self):
        self.project_path = PROJECT_PATH
        self.n_age = None
        self.age_data = None
        self.cm = None
        self.params = None
        self.state_data = None
        self.trans_data = None
        self.tms_data = None

        self.device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
