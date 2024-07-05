import pandas as pd
from pydirac.core.settings import Settings

from pydirac_papertools.formula import calc_pe
from pydirac_papertools.processor.abstract import AtomAbstractDataProcessor
from pydirac_papertools.processor.utils import (
    BEST_CALC_DICT,
    get_error_tex,
    get_tag,
    load_tex_template,
)

__all__ = ["DCCCDataProcessor"]


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

        body_tex = "".join(body_lis)
        return body_tex

    def find_best(self):
        """See ```AbstractDataProcessor.find_best```."""
        s = self.symbol
        k1 = BEST_CALC_DICT[self.METHOD][s][0]
        if not k1:
            return Settings()
        k1 = s + "@" + k1
        if k1 in self.polar:
            return Settings(self.polar[k1])
        else:
            return Settings()
        # return Settings(self.polar[k1])

    def find_best_error(self):
        """See ```AbstractDataProcessor.find_best_error```."""
        s = self.symbol
        k1 = BEST_CALC_DICT[self.METHOD][s][0]
        if not k1:
            return Settings()
        k1 = s + "@" + k1
        if k1 in self.polar:
            return Settings(self.polar_error[k1])
        else:
            return Settings()

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
