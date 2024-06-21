import json
import os.path

from monty.json import MontyDecoder, MSONable
from pydirac.analysis.polarizability import get_polarizability
from pydirac.core.periodic import Element
from pydirac.core.settings import Settings

from pydirac_papertools.constant import get_gs_term
from pydirac_papertools.processor.utils import BEST_CALC_DICT


class AbstractDataProcessor(MSONable):
    """The abstract data processor."""

    KEYS_FOR_SAVE = []
    OPTION_KEYS_FOR_SAVE = []

    def __init__(self):
        for k in self.OPTION_KEYS_FOR_SAVE:
            setattr(self, f"_{k}", None)

    def _assign_values(self):
        data = self.load_data()
        nb = len(self.OPTION_KEYS_FOR_SAVE)
        if nb > 1:
            for i in range(nb):
                setattr(self, f"_{self.OPTION_KEYS_FOR_SAVE[i]}", data[i])
        elif nb == 1:
            setattr(self, f"_{self.OPTION_KEYS_FOR_SAVE[0]}", data)

    def __getattr__(self, item):
        if item in self.OPTION_KEYS_FOR_SAVE:
            if getattr(self, f"_{item}") is None:
                self._assign_values()
            return getattr(self, f"_{item}")
        else:
            return super().__getattribute__(item)

    def as_dict(self) -> dict:
        """Save to a dict."""
        d = {k: getattr(self, k) for k in self.KEYS_FOR_SAVE + self.OPTION_KEYS_FOR_SAVE}
        # d["@module"] = self.__class__.__module__
        # d["@class"] = self.__class__.__name__
        return d

    def to_file(self, filename):
        """Dump object to a json file."""
        json_codes = self.to_json()
        with open(filename, "w") as f:
            f.write(json_codes)

    def to_dataframe(self):
        """Save origin energy to a `DataFrame` object."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """Create an object from a dict."""
        decoded = {
            k: MontyDecoder().process_decoded(v) for k, v in d.items() if not k.startswith("@")
        }
        kwargs = {k: decoded[k] for k in cls.KEYS_FOR_SAVE}
        obj = cls() if type(cls) is AbstractDataProcessor else cls(**kwargs)
        for k in obj.OPTION_KEYS_FOR_SAVE:
            if k in decoded:
                setattr(obj, f"_{k}", decoded[k])
        return obj

    @classmethod
    def from_json(cls, codes):
        """Create an object from json codes."""
        return cls.from_dict(json.loads(codes))

    @classmethod
    def from_file(cls, filename):
        """Create an object from json file."""
        with open(filename) as f:
            json_codes = f.read()
        return cls.from_json(json_codes)

    def to_tex(self):
        """Return latex codes."""
        raise NotImplementedError

    def load_data(self):
        """Load data using pydirac API."""
        raise NotImplementedError

    def __str__(self):
        return self.to_tex()


class AtomAbstractDataProcessor(AbstractDataProcessor):
    METHOD = "empty"
    KEYS_FOR_SAVE = ["dirname", "symbol", "is_quad"]
    KEYS_FOR_SAVE += ["patterns", "threshold"]
    OPTION_KEYS_FOR_SAVE = ["polar", "polar_error", "energy"]

    def __init__(self, dirname, symbol, is_quad=False, patterns=None, threshold=None):
        """
        It should be noted that once another argument is added in initial function, it should also
        be added in KEY_FOR_SAVE.
        """
        self.dirname = os.path.abspath(dirname)
        self.element = Element(symbol)
        self.symbol = self.element.symbol
        self.is_quad = is_quad

        if "DC" in self.METHOD:
            self.gs_term = get_gs_term(self.element.group)
        else:
            self.gs_term = get_gs_term(self.element.group, scheme="LS")
        self.dq = "quadrupole" if self.is_quad else "dipole"
        self.dq_tag = self.dq[0].upper()
        self.patterns = patterns or ["dyall", "ANO-RCC", "faegri"]
        if threshold is None:
            self.threshold = 0.002 if self.element.group in [1] else 0.005
        else:
            self.threshold = threshold

        super().__init__()

    def load_data(self):
        """Load data using pydirac API."""
        data = get_polarizability(
            self.dirname, self.patterns, verbos=False, threshold=self.threshold
        )
        return self.clean_data(data)

    def check_best(self, calc_type):
        """Whether a calculation uses the most expensive parameters.

        Parameters
        ----------
        calc_type
        """
        s = self.symbol
        cases = BEST_CALC_DICT[self.METHOD][s]
        if len(cases):
            for case in cases:
                if calc_type in case:
                    return True
        return False

    def as_dict(self) -> dict:
        """Save to a dict."""
        d = {k: getattr(self, k) for k in self.KEYS_FOR_SAVE + self.OPTION_KEYS_FOR_SAVE}
        # d["@module"] = self.__class__.__module__
        # d["@class"] = self.__class__.__name__
        return d

    def to_file(self, filename):
        """Dump object to a json file."""
        json_codes = self.to_json()
        with open(filename, "w") as f:
            f.write(json_codes)

    @classmethod
    def from_json(cls, codes):
        """Create an object from json codes."""
        return cls.from_dict(json.loads(codes))

    @classmethod
    def from_file(cls, filename):
        """Create an object from json file."""
        with open(filename) as f:
            json_codes = f.read()
        return cls.from_json(json_codes)

    @staticmethod
    def clean_data(data):
        """Clean data obtained from pydirac API."""
        ml_energy = Settings()
        ml_polar = Settings()
        ml_polar_error = Settings()
        if data:
            for k1, v1 in data.items():
                assert k1 in ["curr_dir", "calc_dir", "sub_dir"]
                for k2, v2 in v1.items():
                    ml_polar[k2] = v2["polar"]
                    ml_energy[k2] = v2["energy"]
                    ml_polar_error[k2] = v2["polar_error"]
        return ml_polar, ml_polar_error, ml_energy

    def to_tex(self, has_header=False, precision=2, only_best=False, has_symbol=False):
        """Return latex codes."""
        raise NotImplementedError

    def find_best(self):
        """Find the best solution in all calculations."""
        raise NotImplementedError

    def find_best_error(self):
        """Find the numerical error of the best solution in all calculations."""
        raise NotImplementedError

    def to_dataframe(self, data_type="energy"):
        return getattr(self, f"to_dataframe_{data_type}")()

    def to_dataframe_energy(self):
        """Save origin energy to a `DataFrame` object."""
        raise NotImplementedError

    def to_dataframe_polar(self):
        """Save origin energy to a `DataFrame` object."""
        raise NotImplementedError
