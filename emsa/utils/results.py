"""Consistent result names, including readers for existing CSV exports."""

import os

import numpy as np


def join_result_name(*parts):
    return "_".join(str(part).strip("_") for part in parts if str(part).strip("_"))


def load_result(folder_name, output_name, *parts, **kwargs):
    name = join_result_name(*parts)
    directory = os.path.join(folder_name, output_name)
    path = os.path.join(directory, join_result_name(output_name, name) + ".csv")
    if not os.path.exists(path):
        # Older exports included an empty scenario as an extra underscore.
        legacy_name = f"{output_name}__{name}" if name else f"{output_name}_"
        legacy_path = os.path.join(directory, legacy_name + ".csv")
        if os.path.exists(legacy_path):
            path = legacy_path
    return np.loadtxt(path, **kwargs)
