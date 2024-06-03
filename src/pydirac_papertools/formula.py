__all__ = ["calc_delta", "calc_pe"]


def calc_delta(ml1, ml0, precision, is_quad=False):
    if is_quad:
        return round((ml1 - ml0) / 4.0, precision)
    else:
        return round(ml1 - ml0, precision)


def calc_pe(num, num_ref, precision):
    return round((num - num_ref) / num_ref * 100, precision)
