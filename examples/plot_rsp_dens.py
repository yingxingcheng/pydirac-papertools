import math
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Rebuild the matplotlib font cache
fm._rebuild()

# mpl.rcParams['font.family'] = 'Avenir'
plt.rc("font", family="serif")
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 2


def plot(ax, fname_lis, scale_y=1e-2, symbol=None):
    # r, weights, charge, spherical_decomposition_1, spherical_decomposition_2, ...
    # data_0 = np.loadtxt(fname_0)
    if type(fname_lis) == str:
        fname_lis = [fname_lis]

    data_1 = None
    for fname in fname_lis:
        if data_1 is None:
            data_1 = np.loadtxt(fname, dtype=np.float)
        else:
            data_1 += np.loadtxt(fname, dtype=np.float)
    data_1 /= len(fname_lis)

    def get_data(l, m):
        assert abs(m) <= l
        ang_nb = list(range(-abs(l), abs(l) + 1))
        idx = ang_nb.index(m)
        if l > 0:
            pos = l**2 + idx + 3
        elif l == 0:
            pos = idx + 3
        else:
            raise RuntimeError("pos error")
        return data_1[:, pos]

    def get_integrand(l, y_sph):
        yp = np.sqrt(4 * np.pi / (2 * l + 1)) * y_sph * r ** (l + 2) * -1
        return yp

    r = data_1[:, 0]
    weights = data_1[:, 1]
    print("charge: ", np.dot(weights, data_1[:, 2]))
    print("charge: ", np.dot(weights, get_integrand(0, get_data(0, 0))))
    print("dipole momentum: ", np.dot(data_1[:, 1], get_integrand(1, get_data(1, -1))))

    # red
    color = "#E82229"
    ax.set_xlabel(r"Distance from nucleus [Bohr]", labelpad=10)

    y_scale = round(math.log(scale_y, 10))
    ax.set_ylabel(
        "Integrand of moment integral \n$\\times "
        "10^{{{scale}}}$ [$e/a_0$]".format(**{"scale": y_scale})
    )

    y = get_data(1, -1)
    y = get_integrand(1, y)
    # ax1.plot(r, y_z - y_ref, '-', color=color, label=r'$C_{10}$',markersize=7)
    ax.plot(r, y, "-", color=color, label=r"$C_{10}$", markersize=7)
    # ax1.plot(r, y_ref , '-', color='k', label=r'$C_{10}$',markersize=7)
    if symbol is None:
        symbol = "Null"
    y_max, y_min = max(y), min(y)
    text_r = 15
    text_y = y_min + (y_max - y_min) * 0.5

    ax.text(
        text_r,
        text_y,
        f"{symbol} \n$Q={np.dot(weights, y):.4f}$",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # scale_y = 1e-2
    ticks_y = mticker.FuncFormatter(lambda x, pos: f"{x / scale_y:g}")
    ax.yaxis.set_major_formatter(ticks_y)

    ax.xaxis.set_tick_params(which="major", size=10, width=2, direction="in", top="on")
    ax.xaxis.set_tick_params(which="minor", size=7, width=2, direction="in", top="on")
    ax.yaxis.set_tick_params(which="major", size=10, width=2, direction="in", right="on")
    ax.yaxis.set_tick_params(which="minor", size=7, width=2, direction="in", right="on")

    # ax1.set_ylim(min(y_z - y_ref), max(y_z - y_ref))
    ax.set_xlim(-2, 20)
    # ax1.set_xlim( 0, 0.5)

    ax.legend(loc=0, frameon=False)


def plot_group_1():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    # Create some mock data
    fname_Li = os.path.abspath("./data/Li_3_1_+0.0005.dat")
    fname_Na = os.path.abspath("./data/Na_3_1_+0.0005.dat")
    fname_K = os.path.abspath("./data/K_3_1_+0.0005.dat")
    fname_Rb = os.path.abspath("./data/Rb_3_1_+0.0005.dat")
    fname_Cs = os.path.abspath("./data/Cs_3_1_+0.0005.dat")
    fname_Fr = os.path.abspath("./data/Fr_3_1_+0.0005.dat")

    plot(ax1, fname_Li, scale_y=1e-2, symbol="(a) Li")
    plot(ax2, fname_Na, scale_y=1e-2, symbol="(b) Na")
    plot(ax3, fname_K, scale_y=1e-2, symbol="(c) K")
    plot(ax4, fname_Rb, scale_y=1e-2, symbol="(d) Rb")
    plot(ax5, fname_Cs, scale_y=1e-2, symbol="(e) Cs")
    plot(ax6, fname_Fr, scale_y=1e-2, symbol="(f) Fr")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_1.pdf", bbox_inches="tight")
    plt.show()


def plot_group_2():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    # Create some mock data
    fname_Be = os.path.abspath("./data/Be_1_1_+0.0005.dat")
    fname_Mg = os.path.abspath("./data/Mg_1_1_+0.0005.dat")
    fname_Ca = os.path.abspath("./data/Ca_1_1_+0.0005.dat")
    fname_Sr = os.path.abspath("./data/Sr_1_1_+0.0005.dat")
    fname_Ba = os.path.abspath("./data/Ba_1_1_+0.0005.dat")
    fname_Ra = os.path.abspath("./data/Ra_1_1_+0.0005.dat")

    plot(ax1, fname_Be, scale_y=1e-3, symbol="(a) Ba")
    plot(ax2, fname_Mg, scale_y=1e-3, symbol="(b) Mg")
    plot(ax3, fname_Ca, scale_y=1e-3, symbol="(c) Ca")
    plot(ax4, fname_Sr, scale_y=1e-3, symbol="(d) Sr")
    plot(ax5, fname_Ba, scale_y=1e-3, symbol="(e) Ba")
    plot(ax6, fname_Ra, scale_y=1e-3, symbol="(f) Ra")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_2.pdf", bbox_inches="tight")
    plt.show()


def plot_group_11():
    fig = plt.figure(figsize=(10.24, 8))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    ax3 = plt.subplot2grid((2, 2), (1, 0))
    # ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Create some mock data
    fname_Cu = os.path.abspath("./data/Cu_3_1_+0.0005.dat")
    fname_Ag = os.path.abspath("./data/Ag_3_1_+0.0005.dat")
    fname_Au = os.path.abspath("./data/Au_3_1_+0.0005.dat")

    plot(ax1, fname_Cu, scale_y=1e-3, symbol="(a) Cu")
    plot(ax2, fname_Ag, scale_y=1e-3, symbol="(b) Ag")
    plot(ax3, fname_Au, scale_y=1e-3, symbol="(c) Au")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_11.pdf", bbox_inches="tight")
    plt.show()


def plot_group_12():
    fig = plt.figure(figsize=(10.24, 8))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Create some mock data
    fname_Zn = os.path.abspath("./data/Zn_1_1_+0.0005.dat")
    fname_Cd = os.path.abspath("./data/Cd_1_1_+0.0005.dat")
    fname_Hg = os.path.abspath("./data/Hg_1_1_+0.0005.dat")
    fname_Cn = os.path.abspath("./data/Cn_1_1_+0.0005.dat")

    plot(ax1, fname_Zn, scale_y=1e-3, symbol="(a) Zn")
    plot(ax2, fname_Cd, scale_y=1e-3, symbol="(b) Cd")
    plot(ax3, fname_Hg, scale_y=1e-3, symbol="(c) Hg")
    plot(ax4, fname_Cn, scale_y=1e-3, symbol="(d) Cn")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_12.pdf", bbox_inches="tight")
    plt.show()


def plot_group_13():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    # Create some mock data
    fname_B = os.path.abspath("./data/B_3_1_+0.0005.dat")
    fname_Al = os.path.abspath("./data/Al_3_1_+0.0005.dat")
    fname_Ga = os.path.abspath("./data/Ga_3_1_+0.0005.dat")
    fname_In = os.path.abspath("./data/In_3_1_+0.0005.dat")
    fname_Tl = os.path.abspath("./data/Tl_3_1_+0.0005.dat")
    fname_Nh = os.path.abspath("./data/Nh_3_1_+0.0005.dat")

    plot(ax1, fname_B, scale_y=1e-3, symbol="(a) B")
    plot(ax2, fname_Al, scale_y=1e-3, symbol="(b) Al")
    plot(ax3, fname_Ga, scale_y=1e-3, symbol="(c) Ga")
    plot(ax4, fname_In, scale_y=1e-3, symbol="(d) In")
    plot(ax5, fname_Tl, scale_y=1e-3, symbol="(e) Tl")
    plot(ax6, fname_Nh, scale_y=1e-3, symbol="(f) Nh")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_13.pdf", bbox_inches="tight")
    plt.show()


def plot_group_14():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    # Create some mock data
    fname_C = os.path.abspath("./data/C_1_1_+0.0005.dat")
    fname_Si = os.path.abspath("./data/Si_1_1_+0.0005.dat")
    fname_Ge = os.path.abspath("./data/Ge_1_1_+0.0005.dat")
    fname_Sn = os.path.abspath("./data/Sn_1_1_+0.0005.dat")
    fname_Pb = os.path.abspath("./data/Pb_1_1_+0.0005.dat")
    fname_Fl = os.path.abspath("./data/Fl_1_1_+0.0005.dat")

    plot(ax1, fname_C, scale_y=1e-3, symbol="(a) C ")
    plot(ax2, fname_Si, scale_y=1e-3, symbol="(b) Si")
    plot(ax3, fname_Ge, scale_y=1e-3, symbol="(c) Ge")
    plot(ax4, fname_Sn, scale_y=1e-3, symbol="(d) Sn")
    plot(ax5, fname_Pb, scale_y=1e-3, symbol="(e) Pb")
    plot(ax6, fname_Fl, scale_y=1e-3, symbol="(f) Fl")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_14.pdf", bbox_inches="tight")
    plt.show()


def plot_group_15():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    # Create some mock data
    fname_N = os.path.abspath("./data/N_3_1_+0.0005.dat")
    fname_N_2 = os.path.abspath("./data/N_3_2_+0.0005.dat")
    fname_P = os.path.abspath("./data/P_3_1_+0.0005.dat")
    fname_P_2 = os.path.abspath("./data/P_3_2_+0.0005.dat")
    fname_As = os.path.abspath("./data/As_3_1_+0.0005.dat")
    fname_As_2 = os.path.abspath("./data/As_3_2_+0.0005.dat")
    fname_Sb = os.path.abspath("./data/Sb_3_1_+0.0005.dat")
    fname_Sb_2 = os.path.abspath("./data/Sb_3_2_+0.0005.dat")
    fname_Bi = os.path.abspath("./data/Bi_3_1_+0.0005.dat")
    fname_Bi_2 = os.path.abspath("./data/Bi_3_2_+0.0005.dat")
    fname_Mc = os.path.abspath("./data/Mc_3_1_+0.0005.dat")
    fname_Mc_2 = os.path.abspath("./data/Mc_3_2_+0.0005.dat")

    plot(ax1, [fname_N, fname_N_2], scale_y=1e-3, symbol="(a) N ")
    plot(ax2, [fname_P, fname_P_2], scale_y=1e-3, symbol="(b) P ")
    plot(ax3, [fname_As, fname_As_2], scale_y=1e-3, symbol="(c) As")
    plot(ax4, [fname_Sb, fname_Sb_2], scale_y=1e-3, symbol="(d) Sb")
    plot(ax5, [fname_Bi, fname_Bi_2], scale_y=1e-3, symbol="(e) Bi")
    plot(ax6, [fname_Mc, fname_Mc_2], scale_y=1e-3, symbol="(f) Mc")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_15.pdf", bbox_inches="tight")
    plt.show()


def get_files(symbol, syms, roots):
    files = []
    for sym in syms:
        for root in roots:
            tmp_str = f"./data/{symbol}_{sym}_{root}_+0.0005.dat"
            if os.path.exists(tmp_str):
                files.append(tmp_str)
    return files


def plot_group_16():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    files_O = get_files("O", [1, 2], [1, 2, 3])
    files_S = get_files("S", [1, 2], [1, 2, 3])
    files_Se = get_files("Se", [1, 2], [1, 2, 3])
    files_Te = get_files("Te", [1, 2], [1, 2, 3])
    files_Po = get_files("Po", [1, 2], [1, 2, 3])
    files_Lv = get_files("Lv", [1, 2], [1, 2, 3])

    plot(ax1, files_O, scale_y=1e-3, symbol="(a) O ")
    plot(ax2, files_S, scale_y=1e-3, symbol="(b) S ")
    plot(ax3, files_Se, scale_y=1e-3, symbol="(c) Se")
    plot(ax4, files_Te, scale_y=1e-3, symbol="(d) Te")
    plot(ax5, files_Po, scale_y=1e-3, symbol="(e) Po")
    plot(ax6, files_Lv, scale_y=1e-3, symbol="(f) Lv")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_16.pdf", bbox_inches="tight")
    plt.show()


def plot_group_17():
    fig = plt.figure(figsize=(10.24, 12))

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    files_F = get_files("F", [3], [1, 2])
    files_Cl = get_files("Cl", [3], [1, 2])
    files_Br = get_files("Br", [3], [1, 2])
    files_I = get_files("I", [3], [1, 2])
    files_At = get_files("At", [3], [1, 2])
    files_Ts = get_files("Ts", [3], [1, 2])

    plot(ax1, files_F, scale_y=1e-3, symbol="(a) F ")
    plot(ax2, files_Cl, scale_y=1e-3, symbol="(b) Cl")
    plot(ax3, files_Br, scale_y=1e-3, symbol="(c) Br")
    plot(ax4, files_I, scale_y=1e-3, symbol="(d) I ")
    plot(ax5, files_At, scale_y=1e-3, symbol="(e) At")
    plot(ax6, files_Ts, scale_y=1e-3, symbol="(f) Ts")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_17.pdf", bbox_inches="tight")
    plt.show()


def plot_group_18():
    fig = plt.figure(figsize=(10.24, 16))

    ax1 = plt.subplot2grid((4, 2), (0, 0))
    ax2 = plt.subplot2grid((4, 2), (0, 1))

    ax3 = plt.subplot2grid((4, 2), (1, 0))
    ax4 = plt.subplot2grid((4, 2), (1, 1))

    ax5 = plt.subplot2grid((4, 2), (2, 0))
    ax6 = plt.subplot2grid((4, 2), (2, 1))

    ax7 = plt.subplot2grid((4, 2), (3, 0))

    files_He = get_files("He", [1], [1])
    files_Ne = get_files("Ne", [1], [1])
    files_Ar = get_files("Ar", [1], [1])
    files_Kr = get_files("Kr", [1], [1])
    files_Xe = get_files("Xe", [1], [1])
    files_Rn = get_files("Rn", [1], [1])
    files_Og = get_files("Og", [1], [1])

    plot(ax1, files_He, scale_y=1e-4, symbol="(a) He")
    plot(ax2, files_Ne, scale_y=1e-4, symbol="(b) Ne")
    plot(ax3, files_Ar, scale_y=1e-4, symbol="(c) Ar")
    plot(ax4, files_Kr, scale_y=1e-4, symbol="(d) Kr")
    plot(ax5, files_Xe, scale_y=1e-3, symbol="(e) Xe")
    plot(ax6, files_Rn, scale_y=1e-3, symbol="(f) Rn")
    plot(ax7, files_Og, scale_y=1e-3, symbol="(g) Og")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./figs/momentum_group_18.pdf", bbox_inches="tight")
    plt.show()


def main():
    # plot_group_1()
    # plot_group_2()
    # plot_group_11()
    # plot_group_12()
    # plot_group_13()
    # plot_group_15()
    plot_group_14()
    # plot_group_16()
    # plot_group_17()
    # plot_group_18()


if __name__ == "__main__":
    # 1 bohr radius = 5.29177210903(80)×10−11 m

    main()
