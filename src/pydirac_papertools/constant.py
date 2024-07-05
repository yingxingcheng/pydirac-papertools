__all__ = ["get_atoms_by_group", "get_gs_term", "get_reference_values", "get_sym_root"]


def get_atoms_by_group(group):
    """Get elements by group number."""
    if group == 1:
        atom_list = ["Li", "Na", "K", "Rb", "Cs", "Fr"]
    elif group == 2:
        atom_list = ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"]
    elif group == 11:
        atom_list = ["Cu", "Ag", "Au"]
    elif group == 12:
        atom_list = ["Zn", "Cd", "Hg", "Cn"]
    elif group == 13:
        atom_list = ["B", "Al", "Ga", "In", "Tl", "Nh"]
    elif group == 14:
        atom_list = ["C", "Si", "Ge", "Sn", "Pb", "Fl"]
    elif group == 15:
        atom_list = ["N", "P", "As", "Sb", "Bi", "Mc"]
    elif group == 16:
        atom_list = ["O", "S", "Se", "Te", "Po", "Lv"]
    elif group == 17:
        atom_list = ["F", "Cl", "Br", "I", "At", "Ts"]
    elif group == 18:
        atom_list = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"]
    else:
        raise TypeError(f"group {group} should be integer [1-2, 11-18]!")
    return atom_list


def get_gs_term(group, scheme="jj"):
    """Get ground-state terms for each group in latex format."""
    if group in [1, 11]:
        t = "^2S_{1/2}"
        t_LS = "^2S"
    elif group in [2, 12, 18]:
        t = "^1S_0"
        t_LS = "^1S"
    elif group in [13]:
        t = "^2P_{1/2}"
        t_LS = "^2P"
    elif group in [14]:
        t = "^3P_0"
        t_LS = "^3P"
    elif group in [15]:
        t = "^4S_{3/2}"
        t_LS = "^4S"
    elif group in [16]:
        t = "^3P_2"
        t_LS = "^3P"
    elif group in [17]:
        t = "^2P_{3/2}"
        t_LS = "^2P"
    else:
        raise NotImplementedError(f"Not support for other group {group}")

    if scheme == "jj":
        return t
    else:
        return t_LS


def get_reference_values(group):
    """Reference values in latex format."""
    if group == 1:
        ref = [
            "164.11",
            r"162.7\pm0.5",
            r"289.7\pm0.3",
            r"319.8\pm0.3",
            r"400.9\pm0.7",
            r"317.8\pm2.4",
        ]
    elif group == 2:
        ref = [
            r"37.74\pm0.03",
            r"71.2\pm0.4",
            r"160.8\pm4",
            r"197.2\pm0.2",
            r"272.9\pm10",
            r"246.0\pm4",
        ]
    elif group == 11:
        ref = [r"46.5\pm0.5", r"55.0\pm8", r"36.0\pm3"]
    elif group == 12:
        ref = [r"38.67\pm0.3", r"46\pm2", r"33.91\pm0.34", r"28\pm2"]
    elif group == 13:
        ref = [
            r"20.5\pm1",
            r"57.8\pm1.0",
            r"50.3\pm3",
            r"65\pm4",
            r"50.0\pm2",
            r"29.2\pm2",
        ]
    elif group == 14:
        ref = [
            r"11.3\pm0.2",
            r"37.3\pm 0.7",
            r"40.0\pm1",
            r"53.0\pm6",
            r"47.0\pm3",
            r"31.0\pm4",
        ]
    elif group == 15:
        ref = [r"7.4\pm0.2", r"25.0\pm1", r"30\pm1", r"43\pm2", r"48\pm4", r"71.0\pm20"]
    elif group == 16:
        ref = [
            r"5.3\pm0.2",
            r"19.4\pm0.1",
            r"28.9\pm1.0",
            r"38.0\pm4",
            r"44.0\pm4",
            "--",
        ]
    elif group == 17:
        ref = [
            r"3.74\pm0.08",
            r"14.6\pm0.1",
            r"21.0\pm1",
            r"32.9\pm1.3",
            r"42.0\pm4",
            r"76.0\pm15",
        ]
    elif group == 18:
        ref = [
            "1.38",
            "2.66",
            "11.08",
            r"16.78\pm0.02",
            r"27.32\pm0.2",
            r"35.0\pm2",
            r"58.0\pm6",
        ]
    else:
        raise NotImplementedError(f"Does not support group-{group} elements")

    return [f"${i}$" for i in ref]


def get_sym_root(group, is_Lv=False):
    """Get atomic terms for `group` elements."""
    if is_Lv:
        sym_root_lis = ["sym_1_root_3", "sym_1_root_1", "sym_2_root_1"]
        terms = [
            get_gs_term(group) + ", M_J=0",
            get_gs_term(group) + ", M_J=1",
            get_gs_term(group) + ", M_J=2",
        ]
        return sym_root_lis, terms

    if group in [1, 2, 11, 12, 18]:
        syms = (3,) if group % 2 else (1,)
        nb_roots = (1,)
        terms = [get_gs_term(group)]
    elif group in [13]:
        syms = (3,)
        nb_roots = (3,)
        terms = [get_gs_term(group), "^2P_{3/2}, M_J=1/2", "^2P_{3/2}, M_J=3/2"]
    elif group in [14]:
        syms = (1,)
        nb_roots = (2,)
        terms = [get_gs_term(group)]
    elif group in [15]:
        syms = (3,)
        nb_roots = (2,)
        terms = [get_gs_term(group) + ", M_J=1/2", get_gs_term(group) + ", M_J=3/2"]
    elif group in [16]:
        syms = (1, 2)
        # nb_roots = (3, 2) # this is because \pm 1 state are degenerated
        nb_roots = (2, 1)
        terms = [
            get_gs_term(group) + ", M_J=0",
            get_gs_term(group) + ", M_J=1",
            get_gs_term(group) + ", M_J=2",
        ]
    elif group in [17]:
        syms = (3,)
        nb_roots = (3,)
        terms = [
            get_gs_term(group) + ", M_J=3/2",
            get_gs_term(group) + ", M_J=1/2",
            "^2P_{1/2}, M_J=1/2",
        ]
    else:
        raise RuntimeError(f"Group-{group} elements does not support!")

    sym_root_lis = []
    for sym, nb_root in zip(syms, nb_roots):
        sym_root_lis += [
            f"sym_{s}_root_{r}" for s, r in zip([sym] * nb_root, range(1, nb_root + 1))
        ]
    return sym_root_lis, terms
