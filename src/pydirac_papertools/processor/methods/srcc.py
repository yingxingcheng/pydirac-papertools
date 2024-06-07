from pathlib import Path

import numpy as np
import pandas as pd
from pydirac.analysis.polarizability import get_polarizability
from pydirac.core.settings import Settings

from pydirac_papertools.formula import calc_delta, calc_pe
from pydirac_papertools.processor.abstract import AtomAbstractDataProcessor
from pydirac_papertools.processor.utils import (
    BEST_CALC_DICT,
    get_error_tex,
    get_tag,
    load_tex_template,
)

__all__ = ["SRCCDataProcessor"]


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
                    get_polarizability(
                        str(p), self.patterns, verbos=False, threshold=self.threshold
                    )
                )
                energy["ml0"] = e0
            elif p.is_dir() and p.name == f"{self.SUBDIR_PREFIX}1":
                p1, p1_error, e1 = self.clean_data(
                    get_polarizability(
                        str(p), self.patterns, verbos=False, threshold=self.threshold
                    )
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
        k1 = BEST_CALC_DICT[self.METHOD][s][0] or "None"
        k2 = BEST_CALC_DICT[self.METHOD][s][1] or "None"
        k1 = s + "@" + k1
        k2 = s + "@" + k2
        return Settings(self.polar[k1]), Settings(self.polar[k2])

    def find_best_error(self):
        """See ```AbstractDataProcessor.find_best_error```."""
        s = self.symbol
        k1 = BEST_CALC_DICT[self.METHOD][s][0] or "None"
        k2 = BEST_CALC_DICT[self.METHOD][s][1] or "None"
        k1 = s + "@" + k1
        k2 = s + "@" + k2
        return Settings(self.polar_error[k1]), Settings(self.polar_error[k2])

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
                    if field > self.threshold:
                        continue
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
