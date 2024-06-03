#!/usr/bin/env python

import argparse
import json

best_dict = {
    "NR-CC": {
        "Li": ["D-2C-NR-CC@s-aug-ANO-RCC@(core 3)[vir 229]"],
        "Na": ["D-2C-NR-CC@s-aug-ANO-RCC@(core 11)[vir 287]"],
        "K": ["D-2C-NR-CC@s-aug-ANO-RCC@(core 19)[vir 311]"],
        "Rb": ["D-2C-NR-CC@s-aug-ANO-RCC@(core 27)[vir 247]"],
        "Cs": ["D-2C-NR-CC@s-aug-ANO-RCC@(core 27)[vir 259]"],
        "Fr": ["D-2C-NR-CC@s-aug-ANO-RCC@(core 41)[vir 217]"],
        "Be": ["D-2C-NR-CC@d-aug-dyall.cv4z@(core 4)[vir 266]"],
        "Mg": ["D-2C-NR-CC@d-aug-dyall.cv4z@(core 10)[vir 314]"],
        "Ca": ["D-2C-NR-CC@d-aug-dyall.cv4z@(core 20)[vir 510]"],
        "Sr": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 28)[vir 346]"],
        "Ba": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 28)[vir 398]"],
        "Ra": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 42)[vir 378]"],
        "Cu": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 29)[vir 299]"],
        "Ag": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 29)[vir 355]"],
        "Au": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 43)[vir 381]"],
        "Zn": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 20)[vir 276]"],
        "Cd": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 30)[vir 344]"],
        "Hg": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 44)[vir 362]"],
        "Cn": ["D-2C-NR-CC@s-aug-dyall.cv4z@(core 44)[vir 434]"],
        "B": ["D-2C-NR-CC@dyall.acv4z@(core 5)[vir 251]"],
        "Al": ["D-2C-NR-CC@dyall.acv4z@(core 13)[vir 363]"],
        "Ga": ["D-2C-NR-CC@dyall.acv4z@(core 31)[vir 481]"],
        "In": ["D-2C-NR-CC@dyall.acv4z@(core 49)[vir 553]"],
        "Tl": ["D-2C-NR-CC@dyall.acv4z@(core 45)[vir 309]"],
        "Nh": ["D-2C-NR-CC@dyall.acv4z@(core 45)[vir 305]"],
        "C": ["D-2C-NR-CC@dyall.acv4z@(core 6)[vir 250]"],
        "Si": ["D-2C-NR-CC@dyall.acv4z@(core 14)[vir 362]"],
        "Ge": ["D-2C-NR-CC@dyall.acv4z@(core 32)[vir 480]"],
        "Sn": ["D-2C-NR-CC@dyall.acv4z@(core 50)[vir 552]"],
        "Pb": ["D-2C-NR-CC@dyall.acv4z@(core 46)[vir 308]"],
        "Fl": ["D-2C-NR-CC@dyall.acv4z@(core 52)[vir 302]"],
        "N": ["D-2C-NR-CC@dyall.acv4z@(core 7)[vir 249]"],
        "P": ["D-2C-NR-CC@dyall.acv4z@(core 15)[vir 361]"],
        "As": ["D-2C-NR-CC@dyall.acv4z@(core 33)[vir 479]"],
        "Sb": ["D-2C-NR-CC@dyall.acv4z@(core 51)[vir 551]"],
        "Bi": ["D-2C-NR-CC@dyall.acv4z@(core 47)[vir 307]"],
        "Mc": ["D-2C-NR-CC@dyall.acv4z@(core 47)[vir 303]"],
        "O": ["D-2C-NR-CC@dyall.acv4z@(core 8)[vir 248]"],
        "S": ["D-2C-NR-CC@dyall.acv4z@(core 16)[vir 360]"],
        "Se": ["D-2C-NR-CC@dyall.acv4z@(core 34)[vir 478]"],
        "Te": ["D-2C-NR-CC@dyall.acv4z@(core 52)[vir 550]"],
        "Po": ["D-2C-NR-CC@dyall.acv4z@(core 48)[vir 296]"],
        "Lv": ["D-2C-NR-CC@dyall.acv4z@(core 48)[vir 294]"],
        "F": ["D-2C-NR-CC@dyall.acv4z@(core 9)[vir 247]"],
        "Cl": ["D-2C-NR-CC@dyall.acv4z@(core 17)[vir 359]"],
        "Br": ["D-2C-NR-CC@dyall.acv4z@(core 35)[vir 477]"],
        "I": ["D-2C-NR-CC@dyall.acv4z@(core 53)[vir 549]"],
        "At": ["D-2C-NR-CC@dyall.acv4z@(core 49)[vir 295]"],
        "Ts": ["D-2C-NR-CC@dyall.acv4z@(core 49)[vir 293]"],
        "He": ["D-2C-NR-CC@dyall.acv4z@(core 2)[vir 104]"],
        "Ne": ["D-2C-NR-CC@dyall.acv4z@(core 10)[vir 246]"],
        "Ar": ["D-2C-NR-CC@dyall.acv4z@(core 18)[vir 358]"],
        "Kr": ["D-2C-NR-CC@dyall.acv4z@(core 36)[vir 476]"],
        "Xe": ["D-2C-NR-CC@dyall.acv4z@(core 26)[vir 286]"],
        "Rn": ["D-2C-NR-CC@dyall.acv4z@(core 50)[vir 294]"],
        "Og": ["D-2C-NR-CC@dyall.acv4z@(core 50)[vir 278]"],
    },
    "SR-CC": {
        "Li": [
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 3)[vir 229]",
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 3)[vir 229]",
        ],
        "Na": [
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 11)[vir 287]",
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 11)[vir 287]",
        ],
        "K": [
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 19)[vir 311]",
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 19)[vir 311]",
        ],
        "Rb": [
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 27)[vir 247]",
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 27)[vir 247]",
        ],
        "Cs": [
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 27)[vir 259]",
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 27)[vir 259]",
        ],
        "Fr": [
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 41)[vir 217]",
            "D-2C-SR-CC@s-aug-ANO-RCC@(core 41)[vir 217]",
        ],
        "Be": [
            "D-2C-SR-CC@d-aug-dyall.cv4z@(core 4)[vir 266]",
            "D-2C-SR-CC@d-aug-dyall.cv4z@(core 4)[vir 266]",
        ],
        "Mg": [
            "D-2C-SR-CC@d-aug-dyall.cv4z@(core 10)[vir 314]",
            "D-2C-SR-CC@d-aug-dyall.cv4z@(core 10)[vir 314]",
        ],
        "Ca": [
            "D-2C-SR-CC@d-aug-dyall.cv4z@(core 20)[vir 510]",
            "D-2C-SR-CC@d-aug-dyall.cv4z@(core 20)[vir 510]",
        ],  # (core 20)[vir 346]
        "Sr": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 28)[vir 346]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 28)[vir 346]",
        ],
        "Ba": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 28)[vir 398]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 28)[vir 398]",
        ],
        "Ra": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 42)[vir 378]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 42)[vir 378]",
        ],  # (core 42)[vir 380]
        "Cu": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 29)[vir 299]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 19)[vir 271]",
        ],
        "Ag": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 29)[vir 355]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 29)[vir 355]",
        ],
        "Au": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 43)[vir 381]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 33)[vir 381]",
        ],
        "Zn": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 20)[vir 276]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 20)[vir 276]",
        ],
        "Cd": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 30)[vir 344]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 30)[vir 344]",
        ],
        "Hg": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 44)[vir 362]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 44)[vir 362]",
        ],
        "Cn": [
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 44)[vir 434]",
            "D-2C-SR-CC@s-aug-dyall.cv4z@(core 44)[vir 434]",
        ],  # (core 48)[vir 434]
        "B": [
            "D-2C-SR-CC@dyall.acv4z@(core 5)[vir 251]",
            "D-2C-SR-CC@dyall.acv4z@(core 5)[vir 251]",
        ],
        "Al": [
            "D-2C-SR-CC@dyall.acv4z@(core 13)[vir 363]",
            "D-2C-SR-CC@dyall.acv4z@(core 13)[vir 363]",
        ],
        "Ga": [
            "D-2C-SR-CC@dyall.acv4z@(core 31)[vir 481]",
            "D-2C-SR-CC@dyall.acv4z@(core 13)[vir 203]",
        ],
        "In": [
            "D-2C-SR-CC@dyall.acv4z@(core 49)[vir 553]",
            "D-2C-SR-CC@dyall.acv4z@(core 31)[vir 309]",
        ],
        "Tl": [
            "D-2C-SR-CC@dyall.acv4z@(core 45)[vir 309]",
            "D-2C-SR-CC@dyall.acv4z@(core 35)[vir 309]",
        ],
        "Nh": [
            "D-2C-SR-CC@dyall.acv4z@(core 45)[vir 305]",
            "D-2C-SR-CC@dyall.acv4z@(core 35)[vir 305]",
        ],
        "C": [
            "D-2C-SR-CC@dyall.acv4z@(core 6)[vir 250]",
            "D-2C-SR-CC@dyall.acv4z@(core 6)[vir 250]",
        ],
        "Si": [
            "D-2C-SR-CC@dyall.acv4z@(core 14)[vir 362]",
            "D-2C-SR-CC@dyall.acv4z@(core 14)[vir 362]",
        ],
        "Ge": [
            "D-2C-SR-CC@dyall.acv4z@(core 32)[vir 480]",
            "D-2C-SR-CC@dyall.acv4z@(core 22)[vir 246]",
        ],
        "Sn": [
            "D-2C-SR-CC@dyall.acv4z@(core 50)[vir 552]",
            "D-2C-SR-CC@dyall.acv4z@(core 32)[vir 302]",
        ],
        "Pb": [
            "D-2C-SR-CC@dyall.acv4z@(core 46)[vir 308]",
            "D-2C-SR-CC@dyall.acv4z@(core 46)[vir 308]",
        ],
        "Fl": [
            "D-2C-SR-CC@dyall.acv4z@(core 52)[vir 302]",
            "D-2C-SR-CC@dyall.acv4z@(core 52)[vir 302]",
        ],  # (core 50)[vir 302]
        "N": [
            "D-2C-SR-CC@dyall.acv4z@(core 7)[vir 249]",
            "D-2C-SR-CC@dyall.acv4z@(core 7)[vir 159]",
        ],
        "P": [
            "D-2C-SR-CC@dyall.acv4z@(core 15)[vir 361]",
            "D-2C-SR-CC@dyall.acv4z@(core 15)[vir 185]",
        ],
        "As": [
            "D-2C-SR-CC@dyall.acv4z@(core 33)[vir 479]",
            "D-2C-SR-CC@dyall.acv4z@(core 15)[vir 185]",
        ],
        "Sb": [
            "D-2C-SR-CC@dyall.acv4z@(core 51)[vir 551]",
            "D-2C-SR-CC@dyall.acv4z@(core 15)[vir 227]",
        ],
        "Bi": [
            "D-2C-SR-CC@dyall.acv4z@(core 47)[vir 307]",
            "D-2C-SR-CC@dyall.acv4z@(core 15)[vir 265]",
        ],
        "Mc": [
            "D-2C-SR-CC@dyall.acv4z@(core 47)[vir 303]",
            "D-2C-SR-CC@dyall.acv4z@(core 15)[vir 265]",
        ],
        "O": [
            "D-2C-SR-CC@dyall.acv4z@(core 8)[vir 248]",
            "D-2C-SR-CC@dyall.acv4z@(core 8)[vir 186]",
        ],
        "S": [
            "D-2C-SR-CC@dyall.acv4z@(core 16)[vir 360]",
            "D-2C-SR-CC@dyall.acv4z@(core 16)[vir 184]",
        ],
        "Se": [
            "D-2C-SR-CC@dyall.acv4z@(core 34)[vir 478]",
            "D-2C-SR-CC@dyall.acv4z@(core 16)[vir 184]",
        ],
        "Te": [
            "D-2C-SR-CC@dyall.acv4z@(core 52)[vir 550]",
            "D-2C-SR-CC@dyall.acv4z@(core 16)[vir 216]",
        ],
        "Po": [
            "D-2C-SR-CC@dyall.acv4z@(core 48)[vir 296]",
            "D-2C-SR-CC@dyall.acv4z@(core 16)[vir 262]",
        ],
        "Lv": [
            "D-2C-SR-CC@dyall.acv4z@(core 48)[vir 294]",
            "D-2C-SR-CC@dyall.acv4z@(core 16)[vir 262]",
        ],
        "F": [
            "D-2C-SR-CC@dyall.acv4z@(core 9)[vir 247]",
            "D-2C-SR-CC@dyall.acv4z@(core 9)[vir 183]",
        ],  # (core 9)[vir 115]
        "Cl": [
            "D-2C-SR-CC@dyall.acv4z@(core 17)[vir 359]",
            "D-2C-SR-CC@dyall.acv4z@(core 17)[vir 169]",
        ],
        "Br": [
            "D-2C-SR-CC@dyall.acv4z@(core 35)[vir 477]",
            "D-2C-SR-CC@dyall.acv4z@(core 17)[vir 183]",
        ],
        "I": [
            "D-2C-SR-CC@dyall.acv4z@(core 53)[vir 549]",
            "D-2C-SR-CC@dyall.acv4z@(core 17)[vir 215]",
        ],
        "At": [
            "D-2C-SR-CC@dyall.acv4z@(core 49)[vir 295]",
            "D-2C-SR-CC@dyall.acv4z@(core 17)[vir 261]",
        ],
        "Ts": [
            "D-2C-SR-CC@dyall.acv4z@(core 49)[vir 293]",
            "D-2C-SR-CC@dyall.acv4z@(core 17)[vir 247]",
        ],
        "He": [
            "D-2C-SR-CC@dyall.acv4z@(core 2)[vir 104]",
            "D-2C-SR-CC@dyall.acv4z@(core 2)[vir 104]",
        ],
        "Ne": [
            "D-2C-SR-CC@dyall.acv4z@(core 10)[vir 246]",
            "D-2C-SR-CC@dyall.acv4z@(core 10)[vir 246]",
        ],
        "Ar": [
            "D-2C-SR-CC@dyall.acv4z@(core 18)[vir 358]",
            "D-2C-SR-CC@dyall.acv4z@(core 18)[vir 358]",
        ],
        "Kr": [
            "D-2C-SR-CC@dyall.acv4z@(core 36)[vir 476]",
            "D-2C-SR-CC@dyall.acv4z@(core 36)[vir 476]",
        ],
        "Xe": [
            "D-2C-SR-CC@dyall.acv4z@(core 26)[vir 286]",
            "D-2C-SR-CC@dyall.acv4z@(core 26)[vir 286]",
        ],  # (core 50)[vir 286]
        "Rn": [
            "D-2C-SR-CC@dyall.acv4z@(core 50)[vir 294]",
            "D-2C-SR-CC@dyall.acv4z@(core 50)[vir 294]",
        ],
        "Og": [
            "D-2C-SR-CC@dyall.acv4z@(core 50)[vir 278]",
            "D-2C-SR-CC@dyall.acv4z@(core 50)[vir 278]",
        ],
    },
    "DC-CC": {
        "Li": ["D-4C-DC-CC@s-aug-ANO-RCC@(core 3)[vir 229]"],
        "Na": ["D-4C-DC-CC@s-aug-ANO-RCC@(core 11)[vir 287]"],
        "K": ["D-4C-DC-CC@s-aug-ANO-RCC@(core 19)[vir 311]"],
        "Rb": ["D-4C-DC-CC@s-aug-ANO-RCC@(core 27)[vir 247]"],
        "Cs": ["D-4C-DC-CC@s-aug-ANO-RCC@(core 27)[vir 259]"],
        "Fr": ["D-4C-DC-CC@s-aug-ANO-RCC@(core 41)[vir 217]"],
        "Be": ["D-4C-DC-CC@d-aug-dyall.cv4z@(core 4)[vir 266]"],
        "Mg": ["D-4C-DC-CC@d-aug-dyall.cv4z@(core 10)[vir 314]"],
        "Ca": ["D-4C-DC-CC@d-aug-dyall.cv4z@(core 20)[vir 346]"],
        "Sr": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 28)[vir 346]"],
        "Ba": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 28)[vir 398]"],
        "Ra": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 42)[vir 380]"],
        "Cu": ["D-4C-DC-CC@s-aug-dyall.cv4z@(core 19)[vir 271]"],
        "Ag": ["D-4C-DC-CC@s-aug-dyall.cv4z@(core 29)[vir 355]"],
        "Au": ["D-4C-DC-CC@s-aug-dyall.cv4z@(core 33)[vir 381]"],
        "Zn": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 20)[vir 276]"],
        "Cd": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 30)[vir 344]"],
        "Hg": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 44)[vir 362]"],
        "Cn": ["D-2C-DC-CC@s-aug-dyall.cv4z@(core 48)[vir 434]"],
        "Al": ["D-4C-DC-CC@dyall.acv4z@(core 13)[vir 363]"],
        "Ga": ["D-4C-DC-CC@dyall.acv4z@(core 13)[vir 203]"],
        "In": ["D-2C-DC-CC@dyall.acv4z@(core 31)[vir 309]"],
        "Tl": ["D-4C-DC-CC@dyall.acv4z@(core 35)[vir 309]"],
        "Nh": ["D-4C-DC-CC@dyall.acv4z@(core 35)[vir 305]"],
        "C": ["D-4C-DC-CC@dyall.acv4z@(core 6)[vir 250]"],
        "Si": ["D-2C-DC-CC@dyall.acv4z@(core 14)[vir 362]"],
        "Ge": ["D-2C-DC-CC@dyall.acv4z@(core 22)[vir 246]"],
        "Sn": ["D-2C-DC-CC@dyall.acv4z@(core 32)[vir 302]"],
        "Pb": ["D-2C-DC-CC@dyall.acv4z@(core 46)[vir 308]"],
        "Fl": ["D-2C-DC-CC@dyall.acv4z@(core 50)[vir 302]"],
        "He": ["D-4C-DC-CC@dyall.acv4z@(core 2)[vir 104]"],
        "Ne": ["D-4C-DC-CC@dyall.acv4z@(core 10)[vir 246]"],
        "Ar": ["D-4C-DC-CC@dyall.acv4z@(core 18)[vir 358]"],
        "Kr": ["D-2C-DC-CC@dyall.acv4z@(core 36)[vir 476]"],
        "Xe": ["D-4C-DC-CC@dyall.acv4z@(core 50)[vir 286]"],
        "Rn": ["D-4C-DC-CC@dyall.acv4z@(core 50)[vir 294]"],
        "Og": ["D-4C-DC-CC@dyall.acv4z@(core 50)[vir 278]"],
    },
    "DC-CI": {
        # "Li": ["D-4C-DC-CI@s-aug-ANO-RCC@(core 3)[vir 209]"],
        # "Na": ["D-4C-DC-CI@s-aug-ANO-RCC@(core 9)[vir 245]"],
        # "K": ["D-4C-DC-CI@s-aug-ANO-RCC@(core 19)[vir 199]"],
        # "Rb": ["D-4C-DC-CI@s-aug-ANO-RCC@(core 19)[vir 231]"],
        # "Cs": ["D-4C-DC-CI@s-aug-ANO-RCC@(core 19)[vir 241]"],
        # "Fr": ["D-4C-DC-CI@s-aug-ANO-RCC@(core 19)[vir 187]"],
        # "Be": ["D-4C-DC-CI@dyall.cv4z@(core 4)[vir 132]"],
        # "Mg": ["D-4C-DC-CI@dyall.cv4z@(core 10)[vir 182]"],
        # "Ca": ["D-4C-DC-CI@dyall.cv4z@(core 20)[vir 228]"],
        # "Sr": ["D-4C-DC-CI@dyall.cv4z@(core 20)[vir 214]"],
        # "Ba": ["D-4C-DC-CI@dyall.cv4z@(core 20)[vir 252]"],
        # "Ra": ["D-4C-DC-CI@dyall.cv4z@(core 20)[vir 234]"],
        # "Cu": ["D-4C-DC-CI@dyall.cv4z@(core 11)[vir 187]"],
        # "Ag": ["D-4C-DC-CI@dyall.cv4z@(core 11)[vir 223]"],
        # "Au": ["D-4C-DC-CI@dyall.cv4z@(core 11)[vir 241]"],
        # "Zn": ["D-4C-DC-CI@dyall.cv4z@(core 12)[vir 154]"],
        # "Cd": ["D-4C-DC-CI@dyall.cv4z@(core 12)[vir 200]"],
        # "Hg": ["D-4C-DC-CI@dyall.cv4z@(core 12)[vir 240]"],
        # "Cn": ["D-4C-DC-CI@dyall.cv4z@(core 12)[vir 264]"],
        "B": ["D-4C-DC-CI@dyall.acv4z@(core 5)[vir 251]"],
        "Al": ["D-4C-DC-CI@dyall.acv4z@(core 13)[vir 223]"],
        "Ga": ["D-4C-DC-CI@dyall.acv4z@(core 13)[vir 203]"],
        "In": ["D-4C-DC-CI@dyall.acv4z@(core 13)[vir 237]"],
        "Tl": ["D-4C-DC-CI@dyall.acv4z@(core 13)[vir 277]"],
        "Nh": ["D-4C-DC-CI@dyall.acv4z@(core 13)[vir 271]"],
        # "C": ["D-4C-DC-CI@dyall.acv4z@(core 6)[vir 162]"],
        # "Si": ["D-4C-DC-CI@dyall.acv4z@(core 14)[vir 204]"],
        # "Ge": ["D-4C-DC-CI@dyall.acv4z@(core 14)[vir 186]"],
        # "Sn": ["D-4C-DC-CI@dyall.acv4z@(core 14)[vir 230]"],
        # "Pb": ["D-4C-DC-CI@dyall.acv4z@(core 14)[vir 276]"],
        # "Fl": ["D-4C-DC-CI@dyall.acv4z@(core 14)[vir 270]"],
        "N": ["D-4C-DC-CI@dyall.acv4z@(core 7)[vir 159]"],
        "P": ["D-4C-DC-CI@dyall.acv4z@(core 15)[vir 185]"],
        "As": ["D-4C-DC-CI@dyall.acv4z@(core 15)[vir 185]"],
        "Sb": ["D-4C-DC-CI@dyall.acv4z@(core 15)[vir 227]"],
        "Bi": ["D-4C-DC-CI@dyall.acv4z@(core 15)[vir 265]"],
        "Mc": ["D-4C-DC-CI@dyall.acv4z@(core 15)[vir 265]"],
        "O": ["D-4C-DC-CI@dyall.acv4z@(core 8)[vir 116]"],
        "S": ["D-4C-DC-CI@dyall.acv4z@(core 16)[vir 184]"],
        "Se": ["D-4C-DC-CI@dyall.acv4z@(core 16)[vir 184]"],
        "Te": ["D-4C-DC-CI@dyall.acv4z@(core 16)[vir 216]"],
        "Po": ["D-4C-DC-CI@dyall.acv4z@(core 16)[vir 262]"],
        "Lv": ["D-4C-DC-CI@dyall.acv4z@(core 16)[vir 262]"],
        "F": ["D-4C-DC-CI@dyall.acv4z@(core 9)[vir 115]"],
        "Cl": ["D-4C-DC-CI@dyall.acv4z@(core 17)[vir 169]"],
        "Br": ["D-4C-DC-CI@dyall.acv4z@(core 17)[vir 183]"],
        "I": ["D-4C-DC-CI@dyall.acv4z@(core 17)[vir 215]"],
        "At": ["D-4C-DC-CI@dyall.acv4z@(core 17)[vir 261]"],
        "Ts": ["D-4C-DC-CI@dyall.acv4z@(core 17)[vir 247]"],
        # "He": ["D-4C-DC-CI@dyall.acv4z@(core 2)[vir 60]"],
        # "Ne": ["D-4C-DC-CI@dyall.acv4z@(core 10)[vir 108]"],
        # "Ar": ["D-4C-DC-CI@dyall.acv4z@(core 18)[vir 156]"],
        # "Kr": ["D-4C-DC-CI@dyall.acv4z@(core 18)[vir 182]"],
        # "Xe": ["D-4C-DC-CI@dyall.acv4z@(core 18)[vir 214]"],
        # "Rn": ["D-4C-DC-CI@dyall.acv4z@(core 18)[vir 246]"],
        # "Og": ["D-4C-DC-CI@dyall.acv4z@(core 18)[vir 246]"],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate best solution info in a JSON file.")
    parser.add_argument(
        "--filename",
        type=str,
        default="best_solutions.json",
        help="Output JSON filename",
    )
    parser.add_argument("--indent", type=int, default=2, help="Indentation level for JSON output")
    args = parser.parse_args()

    with open(args.filename, "w") as f:
        json.dump(best_dict, f, indent=args.indent)


if __name__ == "__main__":
    main()
