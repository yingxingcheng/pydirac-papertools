#!/usr/env/bin python

import os

import pandas as pd

from pydirac_papertools.figure import Ploter
from pydirac_papertools.processor import GroupDataProcessor
from pydirac_papertools.table import Tabulator


def make_table_summary(scratch_dir, data):
    """Make paper latex table of summary results."""
    tb = Tabulator(data)
    tex_str = tb.to_latex(add_derived_prop=False)
    texpath = os.path.join(scratch_dir, f"summary_tab_group_{tb.group}.tex")
    with open(texpath, "w") as f:
        f.write(tex_str)


def make_figure(scratch_dir, data):
    """Make paper figures."""
    Ploter(data, save_path=scratch_dir).plot()


def get_old_detail_tex(scratch_dir, dp):
    texs = []
    for atom in dp.OPTION_KEYS_FOR_SAVE:
        adp = getattr(dp, atom)
        texs.append(adp.to_tex(only_best=True))

    with open(os.path.join(scratch_dir, f"group-{gp}.tex"), "w") as f:
        f.write("\n\n".join(texs))


def get_new_detail_tex(scratch_dir, dp):
    with open(os.path.join(scratch_dir, f"group-{gp}.tex"), "w") as f:
        f.write(dp.to_tex(only_best=True))


if __name__ == "__main__":
    data_root = "backup_2020_Sep_21"
    scratch_dir = "newAPI"
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    dfs_energy = []
    dfs_polar = []
    for gp in [1, 2, 11, 12, 13, 14, 15, 16, 17, 18]:
        # for gp in [15]:
        cache_file = os.path.join(scratch_dir, f"group-{gp}.json")
        if os.path.exists(cache_file):
            dp = GroupDataProcessor.from_file(cache_file)
        else:
            dp = GroupDataProcessor(data_root, gp, {"is_quad": False})
            dp.to_file(cache_file)

        df1 = dp.to_dataframe(data_type="energy")
        df2 = dp.to_dataframe(data_type="polar")
        # print(df2)
        # df2.to_csv("tmp.csv")
        # exit()
        dfs_energy.append(df1)
        dfs_polar.append(df2)

        data = dp.analysis()

        make_figure(scratch_dir, data)
        make_table_summary(scratch_dir, data)

        # get_old_detail_tex(scratch_dir, dp)
        get_new_detail_tex(scratch_dir, dp)

        print(f"group {gp} done!")
    df_tot1 = pd.concat(dfs_energy)
    df_tot2 = pd.concat(dfs_polar)
    df_tot1.to_csv(os.path.join(scratch_dir, "energy_data.csv"))
    df_tot2.to_csv(os.path.join(scratch_dir, "polar_data.csv"))
