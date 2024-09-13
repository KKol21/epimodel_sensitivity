from abc import ABC
from os.path import dirname, realpath

PROJECT_PATH = dirname(dirname(dirname(realpath(__file__))))


class DataLoaderBase(ABC):
    def __init__(self):
        self.project_path = PROJECT_PATH
        self.age_data = None
        self.cm = None
        self.params = None
        self.device = None
