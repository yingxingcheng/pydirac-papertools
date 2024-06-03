#!/usr/bin/env python


def generate_data(fin, fout=None, backup=False):
    fout = fout or fin

    with open(fin) as f:
        lines = f.readlines()

    new_lines = []
    for l in lines:
        if l.strip()[0] in ["#", "!"]:
            continue

        field, energy = l.split()
        field = field.split("_")[-1]
        new_lines.append(field + " " + energy)

    if backup:
        with open(fin + ".bak", "w") as f:
            f.write(lines)

    with open(fout, "w") as f:
        f.write("\n".join(new_lines))


if __name__ == "__main__":
    import sys

    fnames = sys.argv[1:]

    for fname in fnames:
        generate_data(fname)
