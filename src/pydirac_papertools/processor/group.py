import pandas as pd

from pydirac_papertools.constant import get_atoms_by_group
from pydirac_papertools.processor.abstract import AbstractDataProcessor
from pydirac_papertools.processor.atom import AtomDataProcessor
from pydirac_papertools.processor.utils import load_tex_template
from pydirac_papertools.table import Tabulator

__all__ = ["GroupDataProcessor"]


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

    def analysis(self, data_type="polar"):
        k_alpha = [f"alpha_{i}" for i in range(3)]
        k_corr = [f"corr_{i}" for i in range(3)]
        k_delta_orb = [f"delta_orb_{i}" for i in [1, "so", 2]]
        k_delta_corr = [f"delta_corr_{i}" for i in [1, "so", 2]]
        k_delta_tot = [f"delta_tot_{i}" for i in [1, "so", 2]]
        k_gs = ["gs_nr", "gs_for_nr", "gs_for_so", "gs_cc_so", "gs_ci_so"]
        keys = k_alpha + k_corr + k_delta_orb + k_delta_corr + k_delta_tot + k_gs

        results = {k: [] for k in keys}
        for atom in self.OPTION_KEYS_FOR_SAVE:
            res = getattr(self, atom).analysis(data_type)
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
            # TODO: here field is not passed
            dfs.append(db.to_dataframe(data_type=data_type))
        df_tot = pd.concat(dfs)
        return df_tot
