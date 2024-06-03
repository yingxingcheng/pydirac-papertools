from pydirac.core.settings import Settings

from pydirac_papertools.processor.methods.srcc import SRCCDataProcessor
from pydirac_papertools.processor.utils import BEST_CALC_DICT

__all__ = ["NRCCDataProcessor"]


class NRCCDataProcessor(SRCCDataProcessor):
    """Non-relativistic Coupled-Cluster data processor."""

    METHOD = "NR-CC"

    def find_best(self):
        """See ```SRCCDataProcessor.find_best```."""
        s = self.symbol
        k1 = BEST_CALC_DICT[self.METHOD][s][0] or "None"
        k1 = s + "@" + k1
        return Settings(self.polar[k1])

    def find_best_error(self):
        """See ```SRCCDataProcessor.find_best```."""
        s = self.symbol
        k1 = BEST_CALC_DICT[self.METHOD][s][0] or "None"
        k1 = s + "@" + k1
        return Settings(self.polar_error[k1])
