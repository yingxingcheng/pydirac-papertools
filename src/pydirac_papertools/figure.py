#!/usr/bin/env python
import importlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["Ploter", "red", "blue"]
data_root = importlib.resources.files("pydirac_papertools.data")

plt.style.use(str(data_root / "plot_style.txt"))
plt.style.use(str(data_root / "plot_style_long_fig.txt"))

warnings.filterwarnings("ignore")

# define colors
red = "#E82229"
blue = "#2E378E"


class Ploter:
    dc_data_style_dict = {
        "color": "k",
        "label": "DC",
        "markerfacecolor": "w",
        "markeredgewidth": 1,
        "markeredgecolor": "k",
    }

    def __init__(self, data, fname=None):
        self.data = data
        self.group = self.data["group"]
        self.atom_list = data["atom_list"]
        self.fname = fname or f"group-{self.group}.png"

        self.fig = plt.figure(figsize=(6, 2.5))
        self.ax1 = plt.subplot2grid((1, 3), (0, 0))
        self.ax2 = plt.subplot2grid((1, 3), (0, 1))
        self.ax3 = plt.subplot2grid((1, 3), (0, 2))
        self.axs = [self.ax1, self.ax2, self.ax3]

    def plot(self, *args, **kwargs):
        """Main plot entry."""
        # self.set_tick()
        self.set_locator_size()

        self.plot_polarizability()
        self.plot_correlation_contribution()
        self.plot_tot_rel()

        self.set_xylim()

        # global plot setup
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(self.fname, bbox_inches="tight")
        # plt.show()

    def plot_polarizability(self, *args, **kwargs):
        """NR-, SR-, DC-polarizabilities"""
        if self.group in [13, 14, 16, 17]:
            # plot for Ml0 and Ml1 of alpha_0
            alpha_0_ml0, alpha_0_ml1, alpha_1_ml0, alpha_1_ml1 = self.get_data_for_ml(
                ml0_callback=lambda x: x.ml0["ccsd(t)"][0],
                ml1_callback=lambda x: x.ml1["ccsd(t)"][0],
            )

            self.ax1.plot(self.atom_list, alpha_0_ml0, "--", color=red, alpha=0.8, label=r"NR0")
            self.ax1.plot(self.atom_list, alpha_0_ml1, "-.", color=red, alpha=0.8, label=r"NR1")
            self.ax1.plot(self.atom_list, self.data["alpha_0"], "-D", color=red, label=r"NR")

            self.ax1.plot(self.atom_list, alpha_1_ml0, "--", color=blue, alpha=0.8, label=r"SR0")
            self.ax1.plot(self.atom_list, alpha_1_ml1, "-.", color=blue, alpha=0.8, label=r"SR1")
            self.ax1.plot(self.atom_list, self.data["alpha_1"], "-s", color=blue, label=r"SR")

            self.ax1.plot(self.atom_list, self.data["alpha_2"], "--o", **self.dc_data_style_dict)

        else:
            self.ax1.plot(self.atom_list, self.data["alpha_0"], "-D", color=red, label=r"NR")
            self.ax1.plot(self.atom_list, self.data["alpha_1"], "-s", color=blue, label=r"SR")
            self.ax1.plot(self.atom_list, self.data["alpha_2"], "--o", **self.dc_data_style_dict)

        self.set_xylabel(self.ax1)
        self.set_legend(self.ax1, 1)

    def plot_correlation_contribution(self, *args, **kwargs):
        """Electron-correlation contribution"""
        self.ax3.plot(self.atom_list, self.data["corr_0"], "-D", color=red, label=r"NR")
        self.ax3.plot(self.atom_list, self.data["corr_1"], "-s", color=blue, label=r"SR")
        if self.group not in [15, 16, 17]:
            self.ax3.plot(self.atom_list, self.data["corr_2"], "--o", **self.dc_data_style_dict)
        else:
            pass
            # The DHF results are computed by average-of-state method, and they cannot be used here.
            # print(self.data.keys())
            # dcscf = np.array([res["scf"][0] for res in self.data["gs_ci_so"]])
            # dcci = np.array([res["gs"] for res in self.data["gs_ci_so"]])
            # corr = dcci - dcscf
            # print("DHF from average-of-state?")
            # print(dcscf)
            # print("DCCISD")
            # print(dcci)
            # self.ax3.plot(self.atom_list, corr, "--o", **self.dc_data_style_dict)

        self.set_xylabel(self.ax3)
        self.ax3.set_ylabel(r"$\Delta \alpha_c$ [a.u.]")
        self.set_legend(self.ax3, 3)

    def plot_tot_rel(self, *args, **kwargs):
        """Relativistic contribution"""
        # atom_Z_list = [Element(e).Z for e in self.atom_list]
        atom_Z_list = self.atom_list
        self.ax2.plot(atom_Z_list, self.data["delta_tot_1"], "-D", color=red, label=r"SR")
        self.ax2.plot(atom_Z_list, self.data["delta_tot_so"], "-s", color=blue, label=r"SOC")
        dc_data = np.asarray(self.data["delta_tot_1"]) + np.asarray(self.data["delta_tot_so"])
        self.ax2.plot(atom_Z_list, dc_data, "--o", **self.dc_data_style_dict)

        self.set_xylabel(self.ax2)
        self.ax2.set_ylabel(r"$\Delta\alpha_r$ [a.u.]")
        self.set_legend(self.ax2, 2)

    def get_data_for_ml(self, ml0_callback, ml1_callback, nr_key="gs_nr", sr_key="gs_for_nr"):
        """data for Ml0 and Ml1 of alpha_0"""
        nr_0_ml0 = [ml0_callback(i) for i in self.data[nr_key]]
        nr_0_ml1 = [ml1_callback(i) for i in self.data[nr_key]]
        sr_1_ml0 = [ml0_callback(i) for i in self.data[sr_key]]
        sr_1_ml1 = [ml1_callback(i) for i in self.data[sr_key]]
        return nr_0_ml0, nr_0_ml1, sr_1_ml0, sr_1_ml1

    def set_xylim(self):
        """Set the axis limits"""
        for i, ax in enumerate(self.axs):
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()

            ylim_l = ylim[1] - ylim[0]
            xlim_l = xlim[1] - xlim[0]
            scale = 0.05
            ylim = (ylim[0] - scale * ylim_l, ylim[1] + scale * ylim_l)
            xlim = (xlim[0] - scale * xlim_l, xlim[1] + scale * xlim_l)

            ylim_l = ylim[1] - ylim[0]
            xlim_l = xlim[1] - xlim[0]

            if self.group in [11]:
                x_center = xlim_l * 0.8 + xlim[0]
                y_center = ylim_l * 0.5 + ylim[0]
            else:
                x_center = xlim_l * 0.1 + xlim[0]
                y_center = ylim_l * 0.5 + ylim[0]

            ax.set_ylim(*ylim)
            ax.set_xlim(*xlim)
            tag = r"(" + "abcd"[i] + ")"
            ax.text(
                x_center,
                y_center,
                tag,
                horizontalalignment="center",
                verticalalignment="center",
            )

    def set_tick(self):
        """Set tick"""
        for ax in self.axs:
            ax.xaxis.set_tick_params(which="major", size=10, width=2, direction="in", top="on")
            ax.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in", top="on")
            ax.yaxis.set_tick_params(which="major", size=10, width=2, direction="in", right="on")
            ax.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in", right="on")

    def set_xylabel(self, ax, use_Z=False):
        """Set label for xy axes."""
        if use_Z:
            ax.set_xlabel(r"Atomic number $Z$")
        else:
            ax.set_xlabel(f"Group {self.group} elements")
        ax.set_ylabel(r"$\alpha$ [a.u.]")

    def set_locator_size(self):
        """Set locator size"""
        ax_nb = len(self.axs)
        if self.group in [1]:
            major_size = [50, 25, 50, 100]
        elif self.group in [2]:
            major_size = [50, 25, 10, 20]
        elif self.group in [11]:
            major_size = [5, 5, 10, 20]
        elif self.group in [12]:
            major_size = [10, 10, 5, 20]
        elif self.group in [13]:
            major_size = [25, 20, 5, 20]
        elif self.group in [14]:
            major_size = [25, 10, 2, 10]
        elif self.group in [15]:
            major_size = [15, 4, 4, 1]
        elif self.group in [16]:
            major_size = [15, 4, 2, 1]
        elif self.group in [17]:
            major_size = [15, 4, 1, 0.5]
        elif self.group in [18]:
            major_size = [15, 4, 0.5, 4]
        else:
            major_size = [50] * ax_nb
        # minor_size = np.asarray(major_size) / 5

        for i, ax in enumerate(self.axs):
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_size[i]))
            # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(minor_size[i]))

    def set_legend(self, ax, nb):
        """Set legend for `nb`-th panel."""
        ax.legend(frameon=False)
        if nb == 1 and self.group in [13, 14, 16, 17]:
            ax.legend(
                loc=0,
                ncol=2,
                borderpad=0.4,
                labelspacing=0.3,
                handlelength=2,
                handletextpad=0.3,
                columnspacing=1,
                frameon=False,
                fontsize=7,
            )
