import numpy as np
import pandas as pd
from pydirac.core.settings import Settings

from pydirac_papertools.processor.abstract import AtomAbstractDataProcessor
from pydirac_papertools.processor.utils import (
    BEST_CALC_DICT,
    get_error_tex,
    get_tag,
    load_tex_template,
)

__all__ = ["DCCIDataProcessor"]


class DCCIDataProcessor(AtomAbstractDataProcessor):
    """MRCI data processor."""

    METHOD = "DC-CI"

    def __str__(self):
        return self.to_tex()

    def to_tex(self, has_header=False, precision=2, only_best=False, has_symbol=False):
        """See ```AbstractDataProcessor.to_tex```."""
        gp = self.element.group
        if gp in [13, 14, 15, 16, 17]:
            body = load_tex_template(f"mrci_group_{gp}_body.tex")
        elif gp in [1, 11]:
            body = load_tex_template("mrci_group_1_11_body.tex")
        elif gp in [2, 12, 18]:
            body = load_tex_template("mrci_group_2_12_18_body.tex")
        else:
            raise RuntimeError(f"Not support group {gp}")

        body_lis = []
        polar_error = self.polar_error
        for k, v in sorted(self.polar.items()):
            ct = k.replace(self.symbol + "@" + self.dq_tag + "-", "")
            if only_best and not self.check_best(ct):
                continue

            if not only_best and self.check_best(ct):
                ct = rf"\textcolor{{cyan}}{{{ct}}}"
                ct = rf"\textbf{{{ct}}}"

            errors = polar_error[k]
            gs = self.get_gs(v, precision)
            scf = self.get_scf(v, precision)
            gs_error = self.get_gs_error(errors, precision, return_tex=True)
            scf_error = self.get_scf_error(errors, precision, return_tex=True)
            if gp in [1, 2, 11, 12, 18]:
                pe_scf = (scf - gs) / gs * 100
                param_dict = {"gs": gs, "calc_type": ct, "scf": scf, "pe_scf": pe_scf}
            else:
                # DHF of DC-CI for p-block elements is computed from average-of-state method.
                param_dict = {"gs": gs, "calc_type": ct, "scf": scf}
            param_dict.update({"gs_error": gs_error, "scf_error": scf_error})

            if gp in [1, 2, 11, 12, 14, 18]:
                # S_{1/2}
                pass
            elif gp in [13, 14, 15, 16, 17]:
                param_dict.update(self.get_extra_states(v, precision))
                param_dict.update(self.get_extra_states_error(errors, precision))
                if gp in [13]:
                    param_dict["diff"] = param_dict["P_3/2"] - param_dict["gs"]
                elif gp in [17]:
                    param_dict["diff"] = param_dict["gs"] - param_dict["P_1/2_1/2"]
            else:
                raise RuntimeError(f"Not support group {gp}")
            body_tex = body.format(**param_dict)
            body_lis.append(body_tex)

        body_tex = "\n".join(body_lis)
        return body_tex

    def get_scf(self, entity, precision=None):
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        scf = entity["scf"][pos]
        scf /= scale
        if precision:
            scf = round(scf, precision)
        return scf

    def get_scf_error(self, entity, precision=None, return_tex=False):
        error = self.get_scf(entity, precision)
        if return_tex:
            return get_error_tex(error)
        else:
            return error

    def get_gs(self, entity, precision=None):
        """Get ground-state results from a CI entity."""
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [1, 11, 13]:
            gs = entity["sym_3_root_1"][pos]
        elif gp in [2, 12, 14, 18]:
            gs = entity["sym_1_root_1"][pos]
        elif gp in [15]:
            gs = np.average([entity["sym_3_root_1"][pos], entity["sym_3_root_2"][pos]])
        elif gp in [16]:
            gs = np.average(
                [
                    entity["sym_1_root_1"][pos],
                    entity["sym_1_root_2"][pos],
                    entity["sym_1_root_3"][pos],
                    entity["sym_2_root_1"][pos],
                    entity["sym_2_root_2"][pos],
                ]
            )
        elif gp in [17]:
            gs = np.average([entity["sym_3_root_1"][pos], entity["sym_3_root_2"][pos]])
        else:
            raise NotImplementedError
        gs /= scale
        if precision:
            gs = round(gs, precision)
        return gs

    def get_gs_error(self, entity, precision=None, return_tex=False):
        """Get ground-state results from a CI entity."""
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [1, 11, 13]:
            gs = entity["sym_3_root_1"][pos]
        elif gp in [2, 12, 14, 18]:
            gs = entity["sym_1_root_1"][pos]
        elif gp in [15]:
            gs = np.sqrt(
                0.5 * entity["sym_3_root_1"][pos] ** 2 + 0.5 * entity["sym_3_root_2"][pos] ** 2
            )
        elif gp in [16]:
            gs = np.sqrt(
                0.2 * entity["sym_1_root_1"][pos] ** 2
                + 0.2 * entity["sym_1_root_2"][pos] ** 2
                + 0.2 * entity["sym_1_root_3"][pos] ** 2
                + 0.2 * entity["sym_2_root_1"][pos] ** 2
                + 0.2 * entity["sym_2_root_2"][pos] ** 2
            )
        elif gp in [17]:
            gs = np.sqrt(
                0.5 * entity["sym_3_root_1"][pos] ** 2 + 0.5 * entity["sym_3_root_2"][pos] ** 2
            )

        else:
            raise NotImplementedError
        gs /= scale
        if precision:
            gs = round(gs, precision)
        if return_tex:
            return get_error_tex(gs)
        else:
            return gs

    def get_extra_states(self, entity, precision=None):
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [13]:
            d = {
                "P_3/2_1/2": entity["sym_3_root_2"][pos],
                "P_3/2_3/2": entity["sym_3_root_3"][pos],
                "P_3/2": np.average([entity["sym_3_root_2"][pos], entity["sym_3_root_3"][pos]]),
            }
        elif gp in [15]:
            d = {
                "P_3/2_1/2": entity["sym_3_root_1"][pos],
                "P_3/2_3/2": entity["sym_3_root_2"][pos],
            }
        elif gp in [16]:
            d = {
                "P_2_0": entity["sym_1_root_1"][pos],
                "P_2_1": np.average([entity["sym_1_root_2"][pos], entity["sym_1_root_3"][pos]]),
                "P_2_2": np.average([entity["sym_2_root_1"][pos], entity["sym_2_root_2"][pos]]),
            }
        elif gp in [17]:
            d = {
                "P_3/2_3/2": entity["sym_3_root_1"][pos],
                "P_3/2_1/2": entity["sym_3_root_2"][pos],
                "P_1/2_1/2": entity["sym_3_root_3"][pos],
            }
        elif gp in [1, 2, 11, 12, 14, 18]:
            d = {}
        else:
            raise NotImplementedError
        d = {k: v / scale for k, v in d.items()}
        if precision:
            d = {k: round(v, precision) for k, v in d.items()}
        return d

    def get_extra_states_error(self, entity, precision=None, return_tex=False):
        pos = 1 if self.is_quad else 0
        scale = 4 if self.is_quad else 1
        gp = self.element.group
        if gp in [13]:
            d = {
                "P_3/2_1/2_error": entity["sym_3_root_2"][pos],
                "P_3/2_3/2_error": entity["sym_3_root_3"][pos],
                "P_3/2_error": np.sqrt(
                    0.5 * entity["sym_3_root_2"][pos] ** 2 + 0.5 * entity["sym_3_root_3"][pos] ** 2
                ),
            }
        elif gp in [15]:
            d = {
                "P_3/2_1/2_error": entity["sym_3_root_1"][pos],
                "P_3/2_3/2_error": entity["sym_3_root_2"][pos],
            }
        elif gp in [16]:
            d = {
                "P_2_0_error": entity["sym_1_root_1"][pos],
                "P_2_1_error": np.sqrt(
                    0.5 * entity["sym_1_root_2"][pos] ** 2 + 0.5 * entity["sym_1_root_3"][pos] ** 2
                ),
                "P_2_2_error": np.sqrt(
                    0.5 * entity["sym_2_root_1"][pos] ** 2 + 0.5 * entity["sym_2_root_2"][pos] ** 2
                ),
            }
        elif gp in [17]:
            d = {
                "P_3/2_3/2_error": entity["sym_3_root_1"][pos],
                "P_3/2_1/2_error": entity["sym_3_root_2"][pos],
                "P_1/2_1/2_error": entity["sym_3_root_3"][pos],
            }
        elif gp in [1, 2, 11, 12, 14, 18]:
            d = {}
        else:
            raise NotImplementedError
        d = {k: v / scale for k, v in d.items()}
        if precision:
            d = {k: round(v, precision) for k, v in d.items()}
        if return_tex:
            d = {k: get_error_tex(v) for k, v in d.items()}
        return d

    def find_best(self):
        """See `AbstractDataProcessor.find_best`."""
        symbol = self.symbol
        if symbol in BEST_CALC_DICT[self.METHOD] and len(BEST_CALC_DICT[self.METHOD][symbol][0]):
            gs_ct = BEST_CALC_DICT[self.METHOD][symbol][0]
            key = symbol + "@" + gs_ct
            gs_res = self.polar[key]
            gs_alpha = self.get_gs(gs_res)
        else:
            gs_alpha = np.nan
            gs_res = Settings()
        gs_res["gs"] = gs_alpha
        return gs_res

    def find_best_error(self):
        """See `AbstractDataProcessor.find_best_error`."""
        symbol = self.symbol
        if symbol in BEST_CALC_DICT[self.METHOD] and len(BEST_CALC_DICT[self.METHOD][symbol][0]):
            gs_ct = BEST_CALC_DICT[self.METHOD][symbol][0]
            key = symbol + "@" + gs_ct
            gs_res = self.polar_error[key]
            gs_alpha = self.get_gs_error(gs_res)
        else:
            gs_alpha = np.nan
            gs_res = Settings()
        gs_res["gs"] = gs_alpha
        return gs_res

    def to_dataframe_energy(self):
        """See ```AbstractDataProcessor.to_dataframe```."""
        data = self.energy
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            fields = entity["fields"]
            energies = entity["energies"]
            for i, field in enumerate(fields):
                d = {"calc_type": ct, "Field": field}
                for state, outs in energies.items():
                    if state == "scf_e" and "SCF" not in d:
                        d["SCF"] = outs[i]
                    else:
                        d["State"] = state
                        d["CISD"] = outs[i]
                d["Tag"] = tags[ct] if ct in tags else 0
                data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar(self):
        pos = 1 if self.is_quad else 0
        data = self.polar
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            p_scf = entity["scf"][pos]
            # ground-state results
            d = {
                "calc_type": ct,
                "State": "gs",
                "SCF": p_scf,
                "CISD": self.get_gs(entity),
            }
            d["Tag"] = tags[ct] if ct in tags else 0
            data_list.append(d)

            # results for extra states
            extra_states = self.get_extra_states(entity)
            for s, v in extra_states.items():
                d = {"calc_type": ct, "State": s, "SCF": p_scf, "CISD": v}
                d["Tag"] = tags[ct] if ct in tags else 0
                data_list.append(d)

            # for state, out in entity.items():
            #     if "scf" not in state:
            #         d = {"calc_type": ct, "State": state, "SCF": p_scf, "CISD": out[pos]}
            #         d["Tag"] = tags[ct] if ct in tags else 0
            #         data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df

    def to_dataframe_polar_error(self):
        pos = 1 if self.is_quad else 0
        data = self.polar_error
        tags = get_tag(method=self.METHOD)
        data_list = []
        for ct in data.keys():
            entity = data[ct]
            p_scf = entity["scf"][pos]
            # ground-state results
            d = {
                "calc_type": ct,
                "State": "gs",
                "SCF": p_scf,
                "CISD": self.get_gs(entity),
            }
            d["Tag"] = tags[ct] if ct in tags else 0
            data_list.append(d)

            # results for extra states
            extra_states = self.get_extra_states(entity)
            for s, v in extra_states.items():
                d = {"calc_type": ct, "State": s, "SCF": p_scf, "CISD": v}
                d["Tag"] = tags[ct] if ct in tags else 0
                data_list.append(d)

            # for state, out in entity.items():
            #     if "scf" not in state:
            #         d = {"calc_type": ct, "State": state, "SCF": p_scf, "CISD": out[pos]}
            #         d["Tag"] = tags[ct] if ct in tags else 0
            #         data_list.append(d)
        df = pd.DataFrame(data_list)
        df.set_index("calc_type", inplace=True)
        return df
