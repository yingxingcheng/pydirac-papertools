from collections import OrderedDict

import numpy as np
import pandas as pd

from pydirac_papertools.constant import *

__all__ = ["Tabulator"]


class Tabulator:
    def __init__(self, data):
        self.data = data
        self.atom_list = self.data["atom_list"]
        self._table = None

    @property
    def group(self):
        """The group of elements."""
        return self.data["group"]

    @property
    def table(self):
        """Make latex table using data generated by processor."""
        if self._table:
            return self._table

        # shorthand
        group = self.group

        # from old API to new one
        old_methods = ["scf", "mp2", "ccsd", "ccsd(t)"]
        new_methods = ["DHF", "MP2", "CCSD", "CCSD(T)"]

        nr_sr_old_keys = ["gs_nr", "gs_for_nr", "gs_for_so"]
        nr_sr_new_keys = ["NR", "NR-SR", "DC-SR"]

        new_data = OrderedDict()
        terms_ls = get_gs_term(group, scheme="LS")
        terms_ls = f"${terms_ls}$"
        if group in [1, 2, 11, 12, 15, 18]:
            ml_comps = ["ml0", "ave"]
            new_ml_comps = ["$M_L=0$", terms_ls]
        elif group in [13, 14, 16, 17]:
            ml_comps = ["ml0", "ml1", "ave"]
            new_ml_comps = ["$M_L=0$", "$M_L=1$", terms_ls]
        else:
            raise NotImplementedError

        # ----------------------
        # copy data from old API
        # ----------------------
        for o, n in zip(nr_sr_old_keys, nr_sr_new_keys):
            for entity in self.data[o]:
                for kk, new_kk in zip(ml_comps, new_ml_comps):
                    for k, new_k in zip(old_methods, new_methods):
                        index = rf"{n}@{new_kk}@{new_k}"

                        if index not in new_data:
                            new_data[index] = []
                        new_data[index].append(entity[kk][k][0])

        # ----------------------
        # copy DC results
        # ----------------------
        gs_term = get_gs_term(group)
        if group in [1, 2, 11, 12, 13, 14, 18]:
            # get gs term
            for entity in self.data["gs_cc_so"]:
                for k, new_k in zip(old_methods, new_methods):
                    index = f"DC@${gs_term}$@{new_k}"

                    if index not in new_data:
                        new_data[index] = []
                    if not entity:
                        new_data[index].append(np.nan)
                    else:
                        new_data[index].append(entity[k][0])
        else:
            for entity in self.data["gs_ci_so"]:
                index = f"DC@${gs_term}$@MRCISD"

                if index not in new_data:
                    new_data[index] = []
                new_data[index].append(entity["gs"])

        # ----------------------
        # copy CI results
        # ----------------------
        # this corresponds the best MRCI specified in best_solution.json.
        if group in [13, 15, 16, 17]:
            for entity in self.data["gs_ci_so"]:
                sym_root_lis, state_terms = get_sym_root(group)
                for sym_root, st in zip(sym_root_lis, state_terms):
                    index = f"DC@${st}$@MRCISD"
                    if index not in new_data:
                        new_data[index] = []
                    new_data[index].append(entity[sym_root][0])

        # -------------------------
        # copy CI excited-state res
        # -------------------------
        group_13_32_ave_index = r"DC@$^2P_{3/2}$@MRCISD"
        group_13_12_gs_index = r"DC@$^2P_{1/2}$@MRCISD"
        group_15_32_32_index = r"DC@$^4S_{3/2}, M_J=3/2$@MRCISD"
        group_15_32_12_index = r"DC@$^4S_{3/2}, M_J=1/2$@MRCISD"
        group_17_32_gs_index = r"DC@$^2P_{3/2}$@MRCISD"
        group_17_12_12_index = r"DC@$^2P_{1/2}, M_J=1/2$@MRCISD"
        if group in [13]:
            for idx in [group_13_32_ave_index]:
                if idx not in new_data:
                    new_data[idx] = []

            state_32_12 = [i["sym_3_root_2"][0] for i in self.data["gs_ci_so"]]
            state_32_32 = [i["sym_3_root_3"][0] for i in self.data["gs_ci_so"]]
            new_data[group_13_32_ave_index] = np.average([state_32_12, state_32_32], axis=0)

        # ----------------------
        # derived results
        # ----------------------
        def get_tex_key(key):
            info = key.split("_")
            if len(info) == 2:
                alpha, order = info[:]
                if alpha == "alpha":
                    k2 = rf"$\alpha^{{({order})}}$"
                else:
                    k2 = rf"$\alpha_{{{alpha}}}^{{({order})}}$"
            elif len(info) == 3:
                d, comp, order = info[:]
                k2 = rf"$\Delta \alpha_{{{comp}}}^{{(\text{{{order}}})}}$"
            else:
                raise RuntimeError(f"Unknown key {key}")
            return k2

        calc_section_key = "Calc."
        # ----------------------
        # obtain Ml0 and Ml1 res
        # ----------------------
        if group in [13, 14, 16, 17]:
            selected_state = terms_ls
            for state in ["$M_L=0$", "$M_L=1$"]:
                nr_dhf = f"NR@{state}@DHF"
                nr_ccsd_t = f"NR@{state}@CCSD(T)"
                sr_nr_dhf = f"NR-SR@{state}@DHF"
                sr_nr_ccsd_t = f"NR-SR@{state}@CCSD(T)"

                tmp_d = OrderedDict()
                tmp_d["alpha_0"] = new_data[nr_ccsd_t]
                tmp_d["alpha_1"] = new_data[sr_nr_ccsd_t]
                tmp_d["corr_0"] = np.asarray(new_data[nr_ccsd_t]) - np.asarray(new_data[nr_dhf])
                tmp_d["corr_1"] = np.asarray(new_data[sr_nr_ccsd_t]) - np.asarray(
                    new_data[sr_nr_dhf]
                )
                tmp_d["delta_corr_1"] = tmp_d["corr_1"] - tmp_d["corr_0"]
                tmp_d["delta_orb_1"] = np.asarray(new_data[sr_nr_dhf]) - np.asarray(
                    new_data[nr_dhf]
                )
                tmp_d["delta_tot_1"] = tmp_d["delta_corr_1"] + tmp_d["delta_orb_1"]

                ml_keys = [f"{calc_section_key}@{state}@" + get_tex_key(k) for k in tmp_d.keys()]
                for k, new_k in zip(tmp_d.keys(), ml_keys):
                    new_data[new_k] = tmp_d[k]
        else:
            selected_state = "$M_L=0$"

        der_keys = [
            "alpha_0",
            "alpha_1",
            "alpha_2",
            "corr_0",
            "corr_1",
            "corr_2",
            "delta_corr_1",
            "delta_corr_so",
            "delta_corr_2",
            "delta_orb_1",
            "delta_orb_so",
            "delta_orb_2",
            "delta_tot_1",
            "delta_tot_so",
            "delta_tot_2",
        ]
        prefix = f"{calc_section_key}@{selected_state}@"
        new_der_keys = [prefix + get_tex_key(k) for k in der_keys]
        for k, new_k in zip(der_keys, new_der_keys):
            new_data[new_k] = self.data[k]

        # add excite-state properties
        if group in [13]:
            delta_32_12 = r"@$\bar{\alpha}_{J=3/2}-\alpha_{J=1/2}$"
            new_data[f"{calc_section_key}@{selected_state}" + delta_32_12] = np.asarray(
                new_data[group_13_32_ave_index]
            ) - np.asarray(new_data[group_13_12_gs_index])
        elif group in [15]:
            delta_32_12 = r"@$\alpha_{J=3/2}-\alpha_{J=1/2}$"
            new_data[f"{calc_section_key}@{selected_state}" + delta_32_12] = np.asarray(
                new_data[group_15_32_32_index]
            ) - np.asarray(new_data[group_15_32_12_index])
        elif group in [17]:
            delta_32_12 = r"@$\bar{\alpha}_{J=3/2}-\alpha_{J=1/2}$"
            new_data[f"{calc_section_key}@{selected_state}" + delta_32_12] = np.asarray(
                new_data[group_17_32_gs_index]
            ) - np.asarray(new_data[group_17_12_12_index])

        self._table = pd.DataFrame.from_dict(
            new_data, orient="index", columns=self.data["atom_list"]
        )
        return self._table

    def to_latex(
        self,
        add_ref=True,
        add_extra_states=True,
        add_derived_prop=True,
        longtable=False,
    ):
        """Generate latex string."""
        table = self.table
        group = self.group

        hams = ["NR", "NR-SR", "DC-SR", "DC"]
        calc_section_key = "Calc."
        methods = ["CCSD(T)", "MRCISD"]

        def _reindex():
            """Reindex the table using multiple indexes of DataFrame."""
            # the new multiple index has three dimension: H, State, Method
            names = [r"$\hat{H}$", "State", "Method"]
            tuples = []
            for i in table.index:
                a, b, c = i.split("@")
                tuples.append((a, b, c))
            table.index = pd.MultiIndex.from_tuples(tuples, names=names)

        _reindex()

        # --------------------------
        # Only write selected data
        # --------------------------
        terms_ls = "${}$".format(get_gs_term(group, scheme="LS"))
        terms_jj = f"${get_gs_term(group)}$"
        states = [terms_ls, terms_jj]

        if add_extra_states:
            # add special cases, i.e., resolved ground-states and simple excited states.
            if group in [1, 2, 11, 12, 15, 18]:
                if group in [15]:
                    # resolved states of ground state
                    group_15_32_12_index = r"$^4S_{3/2}, M_J=1/2$"
                    group_15_32_32_index = r"$^4S_{3/2}, M_J=3/2$"
                    states.extend([group_15_32_12_index, group_15_32_32_index])
            elif group in [13, 14, 15, 16, 17]:
                if group in [13]:
                    # excited state and corresponding resolved states
                    group_13_32_ave_index = r"$^2P_{3/2}$"
                    group_13_32_12_index = r"$^2P_{3/2}, M_J=1/2$"
                    group_13_32_32_index = r"$^2P_{3/2}, M_J=3/2$"
                    states.extend(
                        [
                            group_13_32_ave_index,
                            group_13_32_12_index,
                            group_13_32_32_index,
                        ]
                    )
                if group in [16]:
                    group_16_2_ave_index = r"$^3P_2$"
                    group_16_2_0_index = r"$^3P_2, M_J=0$"
                    group_16_2_1_index = r"$^3P_2, M_J=1$"
                    group_16_2_2_index = r"$^3P_2, M_J=2$"
                    states.extend(
                        [
                            group_16_2_ave_index,
                            group_16_2_0_index,
                            group_16_2_1_index,
                            group_16_2_2_index,
                        ]
                    )
                elif group in [17]:
                    # resolved states of ground state
                    group_17_32_12_index = r"$^2P_{3/2}, M_J=1/2$"
                    group_17_32_32_index = r"$^2P_{3/2}, M_J=3/2$"
                    # excited state
                    group_17_12_12_index = r"$^2P_{1/2}, M_J=1/2$"
                    states.extend(
                        [
                            group_17_32_32_index,
                            group_17_32_12_index,
                            group_17_12_12_index,
                        ]
                    )
            else:
                raise NotImplementedError

        # add alphas
        df_calc = table.loc[(hams, states, methods)]

        if add_ref:
            # add references
            data_ref = get_reference_values(self.group)
            df_ref = pd.DataFrame.from_dict(
                {(r"Ref.~\citenum{Schwerdtfeger2019}", terms_jj, r"$--$"): data_ref},
                orient="index",
                columns=df_calc.columns,
            )
        else:
            df_ref = pd.DataFrame()

        if add_derived_prop:
            # add derived properties
            df_prop = table.loc[[calc_section_key]]
            caption = (
                r"Static dipole polarizabilities with non-relativistic (NR), "
                r"scalar-relativistic (SR), full Dirac-Coulomb (DC) relativistic effects "
                r"of group-{} atoms, as well as their derived properties "
                r"that are defined in Sec. \ref{{sec:deriv_value}}".format(self.group)
            )
        else:
            df_prop = pd.DataFrame()
            caption = (
                r"Static dipole polarizabilities with non-relativistic (NR), "
                r"scalar-relativistic (SR), full Dirac-Coulomb (DC) relativistic effects "
                r"of Group {} elements.".format(self.group)
            )
        df_tot = pd.concat([df_calc, df_ref, df_prop])
        df_tot.drop_duplicates(inplace=True)

        label = rf"tab:dipole_group_{self.group}"
        tex = df_tot.to_latex(
            longtable=longtable,
            float_format="{:0.2f}".format,
            escape=False,
            caption=caption,
            label=label,
            na_rep="$--$",
        )
        # Customize the font size
        if not longtable:
            if group in [1, 2, 12, 18]:
                tex = tex.replace("NR-SR", "SR")
            else:
                tex = tex.replace("NR-SR", r"SR$_n$")
                tex = tex.replace("DC-SR", r"SR$_d$")

            if group in [17]:
                tex = tex.replace(r"\begin{tabular}", r"\scriptsize \begin{tabular}")
            else:
                tex = tex.replace(r"\begin{tabular}", r"\footnotesize \begin{tabular}")
        return tex
