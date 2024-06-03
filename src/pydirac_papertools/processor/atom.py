import os

import numpy as np
import pandas as pd
from pydirac.core.settings import Settings

from pydirac_papertools.processor.abstract import (
    AbstractDataProcessor,
    AtomAbstractDataProcessor,
)
from pydirac_papertools.processor.methods import (
    DCCCDataProcessor,
    DCCIDataProcessor,
    NRCCDataProcessor,
    SRCCDataProcessor,
)
from pydirac_papertools.processor.utils import load_tex_template

__all__ = ["AtomDataProcessor"]


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

    def find_best_error(self):
        """See ```AtomAbstractDataProcessor.find_best_error```."""
        result = Settings()
        for k in self.OPTION_KEYS_FOR_SAVE:
            dp = getattr(self, k)
            if dp:
                result[k] = getattr(self, k).find_best_error()
        return result

    def analysis(self, data_type="polar"):
        """Relativistic effects analysis."""
        if data_type == "polar":
            best_res = self.find_best()
        elif data_type == "polar_error":
            best_res = self.find_best_error()
        else:
            raise RuntimeError(f"Unknown data_type: {data_type}")

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
