import json
import os.path
from pathlib import Path

import importlib_resources
import numpy as np
import pandas as pd
from monty.json import MontyDecoder, MSONable
from pydirac.analysis.polarizability import get_polarizability
from pydirac.core.periodic import Element
from pydirac.core.settings import Settings

from pydirac_papertools.constant import get_atoms_by_group, get_gs_term
from pydirac_papertools.formula import calc_delta, calc_pe
from pydirac_papertools.table import Tabulator

__all__ = [
    "best_dict",
    "AbstractDataProcessor",
    "SRCCDataProcessor",
    "DCCIDataProcessor",
    "DCCCDataProcessor",
    "NRCCDataProcessor",
    "AtomDataProcessor",
    "GroupDataProcessor",
]

best_dict = Settings(
    json.loads(importlib_resources.read_text("pydirac_papertools.data", "best_solutions.json"))
)


def get_error_tex(error):
    return "" if np.isclose(error, 0.0) else rf"\pm {error}"


def get_tag(method):
    # load the best calculations for current `METHOD`
    d = {}
    for k, v in best_dict[method].items():
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
    KEYS_FOR_SAVE += ["patterns"]
    OPTION_KEYS_FOR_SAVE = ["polar", "polar_error", "energy"]

    def __init__(self, dirname, symbol, is_quad=False, patterns=None):
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

        super().__init__()

    def load_data(self):
        """Load data using pydirac API."""
        return self.clean_data(get_polarizability(self.dirname, self.patterns, verbos=False))

    def check_best(self, calc_type):
        """Whether a calculation uses the most expensive parameters.

        Parameters
        ----------
        calc_type
        """
        s = self.symbol
        cases = best_dict[self.METHOD][s]
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

    def to_dataframe(self, data_type="energy"):
        return getattr(self, f"to_dataframe_{data_type}")()

    def to_dataframe_energy(self):
        """Save origin energy to a `DataFrame` object."""
        raise NotImplementedError

    def to_dataframe_polar(self):
        """Save origin energy to a `DataFrame` object."""
        raise NotImplementedError


class SRCCDataProcessor(AtomAbstractDataProcessor):
    """Data processor for scalar-relativistic couple cluster method."""

    METHOD = "SR-CC"
    SUBDIR_PREFIX = "Ml"

    def load_data(self):
        """See ```AbstractDataProcessor.load_data```."""
        p0, e0, p1, e1 = None, None, None, None
        p0_error, p1_error = None, None
        energy = Settings()
        for p in Path(self.dirname).glob(f"{self.SUBDIR_PREFIX}*"):
            if p.is_dir() and p.name == f"{self.SUBDIR_PREFIX}0":
                p0, p0_error, e0 = self.clean_data(
                    get_polarizability(str(p), self.patterns, verbos=False)
                )
                energy["ml0"] = e0
            elif p.is_dir() and p.name == f"{self.SUBDIR_PREFIX}1":
                p1, p1_error, e1 = self.clean_data(
                    get_polarizability(str(p), self.patterns, verbos=False)
                )
                energy["ml1"] = e1

        polar = Settings()
        if p1:
            # we assume that res_ml0 is not None, i.e., both ml0 and ml1 states exist
            f0 = Settings(p0).flatten()
            f1 = Settings(p1).flatten()
            ave = Settings({k: (f0[k] + f1[k] * 2) / 3 for k in f0 if k in f1})
            ave = ave.unflatten()

            for k in ave.keys():
                if k not in polar:
                    polar[k] = Settings()
                polar[k]["ml0"] = p0[k]
                polar[k]["ml1"] = p1[k]
                polar[k]["ave"] = ave[k]
        elif p0:
            ave = p0
            for k in ave.keys():
                if k not in polar:
                    polar[k] = Settings()
                polar[k]["ml0"] = p0[k]
                polar[k]["ave"] = ave[k]
        else:
            raise RuntimeError

        polar_error = Settings()
        if p1_error:
            # we assume that res_ml0 is not None, i.e., both ml0 and ml1 states exist
            f0 = Settings(p0_error).flatten()
            f1 = Settings(p1_error).flatten()
            # ave = Settings({k: (f0[k] + f1[k] * 2) / 3 for k in f0 if k in f1})
            ave = Settings(
                {k: np.sqrt(1 / 3 * f0[k] ** 2 + 2 / 3 * f1[k] ** 2) for k in f0 if k in f1}
            )
            ave = ave.unflatten()

            for k in ave.keys():
                if k not in polar_error:
                    polar_error[k] = Settings()
                polar_error[k]["ml0"] = p0_error[k]
                polar_error[k]["ml1"] = p1_error[k]
                polar_error[k]["ave"] = ave[k]
        elif p0_error:
            ave = p0_error
            for k in ave.keys():
                if k not in polar_error:
                    polar_error[k] = Settings()
                polar_error[k]["ml0"] = p0_error[k]
                polar_error[k]["ave"] = ave[k]
        else:
            raise RuntimeError

        return polar, polar_error, energy

    def has_ml1(self):
        """Whether the state M_l = 1 exists."""
        return self.element.group in [13, 14, 16, 17]

    def to_tex(self, has_header=False, precision=2, only_best=False, has_symbol=False):
        """See ```AbstractDataProcessor.to_tex```."""
        if len(self.polar) < 1:
            return ""

        ele = self.element
        ele_Z = ""
        ele_symbol = "   "
        split_line = "\\cline{3-7}\n         "
        if has_symbol:
            ele_Z = ele.Z
            ele_symbol = ele.symbol
            split_line = ""

        # write body tex
        if self.has_ml1():
            body = load_tex_template("srcc_dipole_body.tex")
            body_q = load_tex_template("srcc_quad_body.tex")
        else:
            body = load_tex_template("srcc_dipole_ml0_body.tex")
            body_q = load_tex_template("srcc_quad_ml0_body.tex")

        body_tex = ""
        polar_error = self.polar_error
        for ct, v in sorted(self.polar.items()):
            if self.is_quad:
                tmp_ct = ct.replace(ele.symbol + "@Q-", "")
                methods = ["scf", "ccsd"]
                body_tmpl = body_q
                pos = 1
            else:
                tmp_ct = ct.replace(ele.symbol + "@D-", "")
                methods = ["scf", "mp2", "ccsd", "ccsd(t)"]
                body_tmpl = body
                pos = 0
            ref_method = methods[-1]

            if only_best and not self.check_best(tmp_ct):
                continue

            if not only_best and self.check_best(tmp_ct):
                if self.METHOD == "SR-CC":
                    tmp_ct = rf"\textcolor{{blue}}{{{tmp_ct}}}"
                else:
                    tmp_ct = rf"\textcolor{{teal}}{{{tmp_ct}}}"
                tmp_ct = rf"\textbf{{{tmp_ct}}}"

            param_dict = {
                "calc_type": tmp_ct,
                "gs_term": self.gs_term,
                "has_line": split_line,
                "ele_Z": ele_Z,
                "ele_symbol": ele_symbol,
            }

            errors = polar_error[ct]
            for ml in v.keys():
                for method in methods:
                    param_dict[f"{ml}_{method}"] = v[ml][method][pos]
                    _error = round(errors[ml][method][pos], precision)
                    param_dict[f"{ml}_{method}_error"] = get_error_tex(_error)
                    if method != ref_method:
                        if ml == "ave":
                            param_dict[f"pe_{method}"] = calc_pe(
                                v[ml][method][pos], v[ml][ref_method][pos], precision
                            )
                        else:
                            param_dict[f"pe_{method}_{ml}"] = calc_pe(
                                v[ml][method][pos], v[ml][ref_method][pos], precision
                            )
                    if ml == "ml1":
                        param_dict[f"delta_{method}"] = calc_delta(
                            v["ml1"][method][pos],
                            v["ml0"][method][pos],
                            precision,
                            self.is_quad,
                        )
            body_tex += body_tmpl.format(**param_dict)

            ele_Z = ""
            ele_symbol = "   "
            split_line = "\\cline{3-7}\n         "
        return body_tex

    def find_best(self):
        """See ```AbstractDataProcessor.find_best```."""
        s = self.symbol
        k1 = best_dict[self.METHOD][s][0] or "None"
        k2 = best_dict[self.METHOD][s][1] or "None"
        k1 = s + "@" + k1
        k2 = s + "@" + k2
        return Settings(self.polar[k1]), Settings(self.polar[k2])

    def to_dataframe_energy(self):
        """See ```AbstractDataProcessor.to_dataframe```."""
        energy = self.energy
        tags = get_tag(method=self.METHOD)
        data_list = []
        for state, data in energy.items():
            for ct in data.keys():
                entity = data[ct]
                fields = entity["fields"]
                energies = entity["energies"]
                for i, field in enumerate(fields):
                    d = {"calc_type": ct, "State": state, "Field": field}
                    d["Tag"] = tags[ct] if ct in tags else 0
                    for method, outs in energies.items():
                        d[method.strip("_e").upper()] = outs[i]
                    data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar(self):
        pos = 1 if self.is_quad else 0
        data = self.polar
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entities = data[ct]
            if self.has_ml1():
                for state, entity in entities.items():
                    d = {"calc_type": ct, "State": "gs" if state == "ave" else state}
                    d["Tag"] = tags[ct] if ct in tags else 0
                    for method, out in entity.items():
                        d[method.upper()] = out[pos]
                    data_list.append(d)
            else:
                # for state, entity in entities.items():
                entity = entities["ave"]
                d = {"calc_type": ct, "State": "gs"}
                d["Tag"] = tags[ct] if ct in tags else 0
                for method, out in entity.items():
                    d[method.upper()] = out[pos]
                data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar_error(self):
        pos = 1 if self.is_quad else 0
        data = self.polar_error
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entities = data[ct]
            if self.has_ml1():
                for state, entity in entities.items():
                    d = {"calc_type": ct, "State": "gs" if state == "ave" else state}
                    d["Tag"] = tags[ct] if ct in tags else 0
                    for method, out in entity.items():
                        d[method.upper()] = out[pos]
                    data_list.append(d)
            else:
                # for state, entity in entities.items():
                entity = entities["ave"]
                d = {"calc_type": ct, "State": "gs"}
                d["Tag"] = tags[ct] if ct in tags else 0
                for method, out in entity.items():
                    d[method.upper()] = out[pos]
                data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df


class NRCCDataProcessor(SRCCDataProcessor):
    """Non-relativistic Coupled-Cluster data processor."""

    METHOD = "NR-CC"

    def find_best(self):
        """See ```SRCCDataProcessor.find_best```."""
        s = self.symbol
        k1 = best_dict[self.METHOD][s][0] or "None"
        k1 = s + "@" + k1
        return Settings(self.polar[k1])


class DCCCDataProcessor(AtomAbstractDataProcessor):
    """Spin-orbital coupling Coupled-Cluster data processor."""

    METHOD = "DC-CC"

    def __str__(self):
        return self.to_tex()

    def to_tex(self, has_header=False, precision=2, only_best=False, has_symbol=False):
        """See ```AbstractDataProcessor.to_tex```."""
        if len(self.polar) < 1:
            return ""

        body = load_tex_template("socc_dipole_body.tex")
        body_d_noT = load_tex_template("socc_dipole_ccsd_body.tex")
        body_q = load_tex_template("socc_quad_ccsd_body.tex")

        if self.is_quad:
            pos, scale = 1, 4
        else:
            pos, scale = 0, 1
        scale = scale

        body_lis = []
        polar_error = self.polar_error
        for ct, v in sorted(self.polar.items()):
            methods = v.keys()

            ref_method = None
            if self.is_quad:
                tmp_ct = ct.replace(self.symbol + "@Q-", "")
                body_tmpl = body_q
                ref_method = "ccsd"
            else:
                if "ccsd(t)" not in methods:
                    if "ccsd" in methods:
                        ref_method = "ccsd"
                    body_tmpl = body_d_noT
                else:
                    ref_method = "ccsd(t)"
                    body_tmpl = body
                tmp_ct = ct.replace(self.symbol + "@D-", "")

            if only_best and not self.check_best(tmp_ct):
                continue

            if not only_best and self.check_best(tmp_ct):
                tmp_ct = rf"\textcolor{{red}}{{{tmp_ct}}}"
                tmp_ct = rf"\textbf{{{tmp_ct}}}"

            param_dict = {
                "calc_type": tmp_ct,
                "gs_term": self.gs_term,
            }

            errors = polar_error[ct]
            for m in methods:
                param_dict[m] = round(v[m][pos], precision)
                _error = round(errors[m][pos], precision)
                param_dict[f"{m}_error"] = get_error_tex(_error)
                if ref_method and m != ref_method:
                    param_dict[f"pe_{m}"] = calc_pe(v[m][pos], v[ref_method][pos], precision)
            body_lis.append(body_tmpl.format(**param_dict))

        body_tex = "\n".join(body_lis)
        return body_tex

    def find_best(self):
        """See ```AbstractDataProcessor.find_best```."""
        s = self.symbol
        k1 = best_dict[self.METHOD][s][0]
        if not k1:
            return Settings()
        k1 = s + "@" + k1
        if k1 in self.polar:
            return Settings(self.polar[k1])
        else:
            return Settings()
        # return Settings(self.polar[k1])

    def to_dataframe_energy(self):
        """See ```AbstractDataProcessor.to_dataframe```."""
        data = self.energy
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            fields = entity["fields"]
            energies = entity["energies"]
            for i, field in enumerate(fields):
                d = {"calc_type": ct, "State": "gs", "Field": field}
                d["Tag"] = tags[ct] if ct in tags else 0
                for method, outs in energies.items():
                    d[method.strip("_e").upper()] = outs[i]
                data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar(self):
        pos = 1 if self.is_quad else 0
        data = self.polar
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            d = {"calc_type": ct, "State": "gs"}
            d["Tag"] = tags[ct] if ct in tags else 0
            for method, out in entity.items():
                d[method.upper()] = out[pos]
            data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar_error(self):
        pos = 1 if self.is_quad else 0
        data = self.polar_error
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            d = {"calc_type": ct, "State": "gs"}
            d["Tag"] = tags[ct] if ct in tags else 0
            for method, out in entity.items():
                d[method.upper()] = out[pos]
            data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df


class DCCIDataProcessor(AtomAbstractDataProcessor):
    """MRCI data processor."""

    METHOD = "DC-CI"

    def __str__(self):
        return self.to_tex()

    def to_tex(self, has_header=False, precision=2, only_best=False, has_symbol=False):
        """See ```AbstractDataProcessor.to_tex```."""
        gp = self.element.group
        if gp in [13, 14, 15, 16, 17]:
            body = load_tex_template(f"mrci_group_{gp}_body.tex")
        elif gp in [1, 11]:
            body = load_tex_template("mrci_group_1_11_body.tex")
        elif gp in [2, 12, 18]:
            body = load_tex_template("mrci_group_2_12_18_body.tex")
        else:
            raise RuntimeError(f"Not support group {gp}")

        body_lis = []
        polar_error = self.polar_error
        for k, v in sorted(self.polar.items()):
            ct = k.replace(self.symbol + "@" + self.dq_tag + "-", "")
            if only_best and not self.check_best(ct):
                continue

            if not only_best and self.check_best(ct):
                ct = rf"\textcolor{{cyan}}{{{ct}}}"
                ct = rf"\textbf{{{ct}}}"

            errors = polar_error[k]
            gs = self.get_gs(v, precision)
            scf = self.get_scf(v, precision)
            gs_error = self.get_gs_error(errors, precision)
            scf_error = self.get_scf_error(errors, precision)
            if gp in [1, 2, 11, 12, 18]:
                pe_scf = (scf - gs) / gs * 100
                param_dict = {"gs": gs, "calc_type": ct, "scf": scf, "pe_scf": pe_scf}
            else:
                # DHF of DC-CI for p-block elements is computed from average-of-state method.
                param_dict = {"gs": gs, "calc_type": ct, "scf": scf}
            param_dict.update({"gs_error": gs_error, "scf_error": scf_error})

            if gp in [1, 2, 11, 12, 14, 18]:
                # S_{1/2}
                pass
            elif gp in [13, 14, 15, 16, 17]:
                param_dict.update(self.get_extra_states(v, precision))
                param_dict.update(self.get_extra_states_error(errors, precision))
                if gp in [13]:
                    param_dict["diff"] = param_dict["P_3/2"] - param_dict["gs"]
                elif gp in [17]:
                    param_dict["diff"] = param_dict["gs"] - param_dict["P_1/2_1/2"]
            else:
                raise RuntimeError(f"Not support group {gp}")
            body_tex = body.format(**param_dict)
            body_lis.append(body_tex)

        body_tex = "\n".join(body_lis)
        return body_tex

    def get_scf(self, entity, precision=None):
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        scf = entity["scf"][pos]
        scf /= scale
        if precision:
            scf = round(scf, precision)
        return scf

    def get_scf_error(self, entity, precision=None):
        error = self.get_scf(entity, precision)
        return get_error_tex(error)

    def get_gs(self, entity, precision=None):
        """Get ground-state results from a CI entity."""
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [1, 11, 13]:
            gs = entity["sym_3_root_1"][pos]
        elif gp in [2, 12, 14, 18]:
            gs = entity["sym_1_root_1"][pos]
        elif gp in [15]:
            gs = np.average([entity["sym_3_root_1"][pos], entity["sym_3_root_2"][pos]])
        elif gp in [16]:
            gs = np.average(
                [
                    entity["sym_1_root_1"][pos],
                    entity["sym_1_root_2"][pos],
                    entity["sym_1_root_3"][pos],
                    entity["sym_2_root_1"][pos],
                    entity["sym_2_root_2"][pos],
                ]
            )
        elif gp in [17]:
            gs = np.average([entity["sym_3_root_1"][pos], entity["sym_3_root_2"][pos]])
        else:
            raise NotImplementedError
        gs /= scale
        if precision:
            gs = round(gs, precision)
        return gs

    def get_gs_error(self, entity, precision=None):
        """Get ground-state results from a CI entity."""
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [1, 11, 13]:
            gs = entity["sym_3_root_1"][pos]
        elif gp in [2, 12, 14, 18]:
            gs = entity["sym_1_root_1"][pos]
        elif gp in [15]:
            gs = np.sqrt(
                0.5 * entity["sym_3_root_1"][pos] ** 2 + 0.5 * entity["sym_3_root_2"][pos] ** 2
            )
        elif gp in [16]:
            gs = np.sqrt(
                0.2 * entity["sym_1_root_1"][pos] ** 2
                + 0.2 * entity["sym_1_root_2"][pos] ** 2
                + 0.2 * entity["sym_1_root_3"][pos] ** 2
                + 0.2 * entity["sym_2_root_1"][pos] ** 2
                + 0.2 * entity["sym_2_root_2"][pos] ** 2
            )
        elif gp in [17]:
            gs = np.sqrt(
                0.5 * entity["sym_3_root_1"][pos] ** 2 + 0.5 * entity["sym_3_root_2"][pos] ** 2
            )

        else:
            raise NotImplementedError
        gs /= scale
        if precision:
            gs = round(gs, precision)
        return get_error_tex(gs)

    def get_extra_states(self, entity, precision=None):
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [13]:
            d = {
                "P_3/2_1/2": entity["sym_3_root_2"][pos],
                "P_3/2_3/2": entity["sym_3_root_3"][pos],
                "P_3/2": np.average([entity["sym_3_root_2"][pos], entity["sym_3_root_3"][pos]]),
            }
        elif gp in [15]:
            d = {
                "P_3/2_1/2": entity["sym_3_root_1"][pos],
                "P_3/2_3/2": entity["sym_3_root_2"][pos],
            }
        elif gp in [16]:
            d = {
                "P_2_0": entity["sym_1_root_1"][pos],
                "P_2_1": np.average([entity["sym_1_root_2"][pos], entity["sym_1_root_3"][pos]]),
                "P_2_2": np.average([entity["sym_2_root_1"][pos], entity["sym_2_root_2"][pos]]),
            }
        elif gp in [17]:
            d = {
                "P_3/2_3/2": entity["sym_3_root_1"][pos],
                "P_3/2_1/2": entity["sym_3_root_2"][pos],
                "P_1/2_1/2": entity["sym_3_root_3"][pos],
            }
        elif gp in [1, 2, 11, 12, 14, 18]:
            d = {}
        else:
            raise NotImplementedError
        d = {k: v / scale for k, v in d.items()}
        if precision:
            d = {k: round(v, precision) for k, v in d.items()}
        return d

    def get_extra_states_error(self, entity, precision=None):
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [13]:
            d = {
                "P_3/2_1/2_error": entity["sym_3_root_2"][pos],
                "P_3/2_3/2_error": entity["sym_3_root_3"][pos],
                "P_3/2_error": np.sqrt(
                    0.5 * entity["sym_3_root_2"][pos] ** 2 + 0.5 * entity["sym_3_root_3"][pos] ** 2
                ),
            }
        elif gp in [15]:
            d = {
                "P_3/2_1/2_error": entity["sym_3_root_1"][pos],
                "P_3/2_3/2_error": entity["sym_3_root_2"][pos],
            }
        elif gp in [16]:
            d = {
                "P_2_0_error": entity["sym_1_root_1"][pos],
                "P_2_1_error": np.sqrt(
                    0.5 * entity["sym_1_root_2"][pos] ** 2 + 0.5 * entity["sym_1_root_3"][pos] ** 2
                ),
                "P_2_2_error": np.sqrt(
                    0.5 * entity["sym_2_root_1"][pos] ** 2 + 0.5 * entity["sym_2_root_2"][pos] ** 2
                ),
            }
        elif gp in [17]:
            d = {
                "P_3/2_3/2_error": entity["sym_3_root_1"][pos],
                "P_3/2_1/2_error": entity["sym_3_root_2"][pos],
                "P_1/2_1/2_error": entity["sym_3_root_3"][pos],
            }
        elif gp in [1, 2, 11, 12, 14, 18]:
            d = {}
        else:
            raise NotImplementedError
        d = {k: v / scale for k, v in d.items()}
        if precision:
            d = {k: round(v, precision) for k, v in d.items()}
        d = {k: get_error_tex(v) for k, v in d.items()}
        return d

    def find_best(self):
        """See `AbstractDataProcessor.find_best`."""
        symbol = self.symbol
        if symbol in best_dict[self.METHOD] and len(best_dict[self.METHOD][symbol][0]):
            gs_ct = best_dict[self.METHOD][symbol][0]
            key = symbol + "@" + gs_ct
            gs_res = self.polar[key]
            gs_alpha = self.get_gs(gs_res)
        else:
            gs_alpha = np.nan
            gs_res = Settings()
        gs_res["gs"] = gs_alpha
        return gs_res

    def to_dataframe_energy(self):
        """See ```AbstractDataProcessor.to_dataframe```."""
        data = self.energy
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            fields = entity["fields"]
            energies = entity["energies"]
            for i, field in enumerate(fields):
                d = {"calc_type": ct, "Field": field}
                for state, outs in energies.items():
                    if state == "scf_e" and "SCF" not in d:
                        d["SCF"] = outs[i]
                    else:
                        d["State"] = state
                        d["CISD"] = outs[i]
                d["Tag"] = tags[ct] if ct in tags else 0
                data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar(self):
        pos = 1 if self.is_quad else 0
        data = self.polar
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            p_scf = entity["scf"][pos]
            # ground-state results
            d = {
                "calc_type": ct,
                "State": "gs",
                "SCF": p_scf,
                "CISD": self.get_gs(entity),
            }
            d["Tag"] = tags[ct] if ct in tags else 0
            data_list.append(d)

            # results for extra states
            extra_states = self.get_extra_states(entity)
            for s, v in extra_states.items():
                d = {"calc_type": ct, "State": s, "SCF": p_scf, "CISD": v}
                d["Tag"] = tags[ct] if ct in tags else 0
                data_list.append(d)

            # for state, out in entity.items():
            #     if "scf" not in state:
            #         d = {"calc_type": ct, "State": state, "SCF": p_scf, "CISD": out[pos]}
            #         d["Tag"] = tags[ct] if ct in tags else 0
            #         data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar_error(self):
        pos = 1 if self.is_quad else 0
        data = self.polar_error
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            p_scf = entity["scf"][pos]
            # ground-state results
            d = {
                "calc_type": ct,
                "State": "gs",
                "SCF": p_scf,
                "CISD": self.get_gs(entity),
            }
            d["Tag"] = tags[ct] if ct in tags else 0
            data_list.append(d)

            # results for extra states
            extra_states = self.get_extra_states(entity)
            for s, v in extra_states.items():
                d = {"calc_type": ct, "State": s, "SCF": p_scf, "CISD": v}
                d["Tag"] = tags[ct] if ct in tags else 0
                data_list.append(d)

            # for state, out in entity.items():
            #     if "scf" not in state:
            #         d = {"calc_type": ct, "State": state, "SCF": p_scf, "CISD": out[pos]}
            #         d["Tag"] = tags[ct] if ct in tags else 0
            #         data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df


class AtomDataProcessor(AtomAbstractDataProcessor):
    """Atom data analysis including different type data processor."""

    OPTION_KEYS_FOR_SAVE = ["nrcc", "srcc", "dccc", "dcci"]
    SUBDIR_NAMES_DIPOLE = ["_nr", "", "_so", "_mrci"]
    SUBDIR_NAMES_QUAD = ["_q_nr", "_q_theta", "_q_so", "_q_mrci"]

    def load_data(self):
        """See ```AbstractDataProcessor.load_data```."""
        s = self.symbol
        names = self.SUBDIR_NAMES_QUAD if self.is_quad else self.SUBDIR_NAMES_DIPOLE
        names = [s + n for n in names]
        dp_cls = [
            NRCCDataProcessor,
            SRCCDataProcessor,
            DCCCDataProcessor,
            DCCIDataProcessor,
        ]
        # dp_cls = getattr(__name__, k.upper() + "DataProcessor") for k in self.OPTION_KEYS_FOR_SAVE

        dps = []
        for sub_path, dp in zip(names, dp_cls):
            param = {k: getattr(self, k) for k in self.KEYS_FOR_SAVE}
            sub_path = os.path.join(self.dirname, sub_path)
            if not os.path.exists(sub_path) or os.path.exists(os.path.join(sub_path, "cannot")):
                _dp = Settings()
            else:
                param["dirname"] = sub_path
                _dp = dp(**param)
            dps.append(_dp)
        return dps

    def to_tex(self, has_header=True, precision=2, only_best=False, has_symbol=True):
        """See ```AtomAbstractDataProcessor.to_tex```."""
        header_tex = ""
        footer_tex = ""
        if has_header:
            header_tmpl = load_tex_template("atom_dipole_header.tex")
            header_tex = header_tmpl.format(
                **{"DQ": self.dq, "dq_tag": self.dq_tag, "ele_type": self.symbol}
            )
            footer_tmpl = load_tex_template("atom_dipole_footer.tex")
            footer_tex = footer_tmpl.format(**{"DQ": self.dq, "ele_type": self.symbol})

        texs = [header_tex]
        write_symbol = has_symbol
        for k in self.OPTION_KEYS_FOR_SAVE:
            dp = getattr(self, k)
            if dp:
                if write_symbol:
                    texs.append(dp.to_tex(False, precision, only_best, True))
                    write_symbol = False
                else:
                    texs.append(dp.to_tex(False, precision, only_best, False))
        texs.append(footer_tex)
        return "".join(texs)

    def find_best(self):
        """See ```AtomAbstractDataProcessor.find_best```."""
        result = Settings()
        for k in self.OPTION_KEYS_FOR_SAVE:
            dp = getattr(self, k)
            if dp:
                result[k] = getattr(self, k).find_best()
        return result

    def analysis(self):
        """Relativistic effects analysis."""
        best_res = self.find_best()
        gs_nr = best_res["nrcc"]
        gs_for_nr, gs_for_so = best_res["srcc"]
        gs_cc_so = best_res["dccc"]
        gs_ci_so = best_res["dcci"]

        corr_0 = gs_nr.ave["ccsd(t)"][0] - gs_nr.ave.scf[0]
        corr_1 = gs_for_nr.ave["ccsd(t)"][0] - gs_for_nr.ave.scf[0]

        alpha_0 = gs_nr.ave["ccsd(t)"][0]
        alpha_1 = gs_for_nr.ave["ccsd(t)"][0]

        if gs_cc_so:
            corr_2 = gs_cc_so["ccsd(t)"][0] - gs_cc_so.scf[0]
            delta_orb_so = gs_cc_so.scf[0] - gs_for_so.ave.scf[0]
            delta_tot_so = gs_cc_so["ccsd(t)"][0] - gs_for_so.ave["ccsd(t)"][0]
            alpha_2 = gs_cc_so["ccsd(t)"][0]
        else:
            corr_2 = np.nan
            delta_orb_so = np.nan
            delta_tot_so = gs_ci_so.gs - gs_for_so.ave["ccsd(t)"][0]
            alpha_2 = gs_ci_so.gs

        delta_orb_1 = gs_for_nr.ave.scf[0] - gs_nr.ave.scf[0]
        delta_orb_2 = delta_orb_1 + delta_orb_so

        delta_corr_1 = corr_1 - corr_0
        delta_corr_so = corr_2 - corr_1
        delta_corr_2 = delta_corr_1 + delta_corr_so

        delta_tot_1 = gs_for_nr.ave["ccsd(t)"][0] - gs_nr.ave["ccsd(t)"][0]
        delta_tot_2 = delta_tot_1 + delta_tot_so

        res_alpha = [alpha_0, alpha_1, alpha_2]
        res_corr = [corr_0, corr_1, corr_2]
        res_delta_orb = [delta_orb_1, delta_orb_so, delta_orb_2]
        res_delta_corr = [delta_corr_1, delta_corr_so, delta_corr_2]
        res_delta_tot = [delta_tot_1, delta_tot_so, delta_tot_2]
        res_gs = [gs_nr, gs_for_nr, gs_for_so, gs_cc_so, gs_ci_so]
        return res_alpha + res_corr + res_delta_orb + res_delta_corr + res_delta_tot + res_gs

    def to_dataframe(self, data_type="energy"):
        """See ```AbstractDataProcessor.to_dataframe```."""
        dfs = []
        for k in self.OPTION_KEYS_FOR_SAVE:
            db = getattr(self, k)
            if isinstance(db, AbstractDataProcessor):
                if data_type == "energy":
                    dfs.append(db.to_dataframe())
                elif data_type == "polar":
                    dfs.append(db.to_dataframe_polar())
                elif data_type == "polar_error":
                    dfs.append(db.to_dataframe_polar_error())
                else:
                    raise TypeError
        table = pd.concat(dfs)

        def _reindex():
            """Reindex the table using multiple indexes of DataFrame."""
            # the new multiple index has three dimension: H, State, Method
            names = [
                "Group",
                "Z",
                "Symbol",
                "Property",
                "Rank of Hamilton",
                "Rank of Relativity",
                "Module",
                "Basis",
                "Correlation Level",
            ]
            tuples = []
            for i in table.index:
                a, calc_info, c, d = i.split("@")
                try:
                    prop, level_h, level_rel, module = calc_info.split("-")
                except ValueError:
                    prop, level_h, level_rel, gaunt, module = calc_info.split("-")
                    level_rel += f"-{gaunt}"

                tuples.append(
                    (
                        self.element.group,
                        self.element.Z,
                        a,
                        prop,
                        level_h,
                        level_rel,
                        module,
                        c,
                        d,
                    )
                )
            table.index = pd.MultiIndex.from_tuples(tuples, names=names)

        # _reindex()
        return table

    def to_dataframe_energy(self):
        return self.to_dataframe("energy")

    def to_dataframe_polar(self):
        return self.to_dataframe("polar")

    def to_dataframe_polar_error(self):
        return self.to_dataframe("polar_error")


class GroupDataProcessor(AbstractDataProcessor):
    """Data processor for a list of elements belong to a same group."""

    KEYS_FOR_SAVE = ["dirname", "group", "atom_param_dict"]

    def __init__(self, dirname, group, atom_param_dict=None):
        self.dirname = dirname
        self.group = group
        self.atom_param_dict = atom_param_dict or {}
        self.OPTION_KEYS_FOR_SAVE = get_atoms_by_group(group)
        super().__init__()

    def load_data(self):
        """See ```AbstractDataProcessor.load_data```."""
        adps = []
        for atom in self.OPTION_KEYS_FOR_SAVE:
            param = {k: v for k, v in self.atom_param_dict.items()}
            param["dirname"] = self.dirname
            param["symbol"] = atom
            adp = AtomDataProcessor(**param)
            adps.append(adp)
        return adps

    def to_tex(self, has_header=True, precision=2, only_best=False, has_symbol=True):
        """See ```AbstractDataProcessor.to_tex```."""
        texs = []
        dq, dq_tag = None, None
        for adp in [getattr(self, k) for k in self.OPTION_KEYS_FOR_SAVE]:
            if dq is None:
                dq = adp.dq
                dq_tag = adp.dq_tag
            texs.append(adp.to_tex(False, precision, only_best, has_symbol))

        header_tex = footer_tex = ""
        if has_header:
            header_tmpl = load_tex_template("group_dipole_header.tex")
            header_tex = header_tmpl.format(**{"DQ": dq, "dq_tag": dq_tag, "gp": self.group})
            footer_tmpl = load_tex_template("group_dipole_footer.tex")
            footer_tex = footer_tmpl.format(**{"DQ": dq, "gp": self.group})
        tmp_tex = "        \\hline\n".join(texs)
        texs = [header_tex] + [tmp_tex] + [footer_tex]
        return "".join(texs)

    def analysis(self):
        k_alpha = [f"alpha_{i}" for i in range(3)]
        k_corr = [f"corr_{i}" for i in range(3)]
        k_delta_orb = [f"delta_orb_{i}" for i in [1, "so", 2]]
        k_delta_corr = [f"delta_corr_{i}" for i in [1, "so", 2]]
        k_delta_tot = [f"delta_tot_{i}" for i in [1, "so", 2]]
        k_gs = ["gs_nr", "gs_for_nr", "gs_for_so", "gs_cc_so", "gs_ci_so"]
        keys = k_alpha + k_corr + k_delta_orb + k_delta_corr + k_delta_tot + k_gs

        results = {k: [] for k in keys}
        for atom in self.OPTION_KEYS_FOR_SAVE:
            res = getattr(self, atom).analysis()
            for i, k in enumerate(keys):
                results[k].append(res[i])
        results["atom_list"] = self.OPTION_KEYS_FOR_SAVE
        results["group"] = self.group
        return results

    def summary(self):
        results = self.analysis()
        tb = Tabulator(results)
        return tb.to_latex()

    def to_dataframe(self, data_type="energy"):
        dfs = []
        for k in self.OPTION_KEYS_FOR_SAVE:
            db = getattr(self, k)
            dfs.append(db.to_dataframe(data_type=data_type))
        df_tot = pd.concat(dfs)
        return df_tot
