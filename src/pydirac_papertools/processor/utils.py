import json
import os.path

import importlib_resources
import numpy as np
from pydirac.core.settings import Settings

__all__ = ["BEST_CALC_DICT", "get_tag", "get_error_tex", "load_tex_template"]


BEST_CALC_DICT = Settings(
    json.loads(importlib_resources.read_text("pydirac_papertools.data", "best_solutions.json"))
)


def get_error_tex(error):
    return "" if np.isclose(error, 0.0) else rf"\pm {error}"


def get_tag(method):
    # load the best calculations for current `METHOD`
    d = {}
    for k, v in BEST_CALC_DICT[method].items():
        for i, ct in enumerate(v):
            key = f"{k}@{ct}"
            if method == "SR-CC":
                d[key] = 2 * (i + 1)
            elif method == "NR-CC":
                d[key] = 1
            elif method == "DC-CC":
                d[key] = 8
            elif method == "DC-CI":
                d[key] = 16
            else:
                raise NotImplementedError
    return d


def load_tex_template(filename):
    """Load tex template for latex output."""
    static_path = str(importlib_resources.files("pydirac_papertools.data"))
    fullpath = os.path.join(static_path, filename)
    tex = ""
    if os.path.exists(fullpath):
        with open(fullpath) as f:
            tex = f.read()
    return tex
