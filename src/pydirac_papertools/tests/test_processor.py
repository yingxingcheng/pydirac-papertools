import json

from pydirac_papertools.constant import get_atoms_by_group
from pydirac_papertools.processor import *
from pydirac_papertools.table import Tabulator


def test_NRCCDataProcessor():
    np = NRCCDataProcessor("backup_2020_Sep_21/Al_nr", "Al")
    print(np.find_best())


def test_SRCCDataProcessor():
    # sp = SRCCDataProcessor("backup_2020_Sep_21/Al_q_theta", "Al", is_quad=True, has_header=True)
    # print(sp)
    # print(sp.as_dict())

    sp_d = SRCCDataProcessor(
        "backup_2020_Sep_21/Al", "Al", is_quad=False, has_header=True, only_best=True
    )

    # print(dict(sp_d))

    # print(sp_d)
    print(sp_d.as_dict())

    # res = sp_d.find_best()
    # print(res)

    # sp = SRCCDataProcessor("backup_2020_Sep_21/He_q_theta", "He", is_quad=True, has_header=True)
    # print(sp)

    # sp = SRCCDataProcessor("backup_2020_Sep_21/He", "He", is_quad=False, has_header=True)
    # print(sp)


def test_SOCCDataProcessor():
    # sp = SOCCDataProcessor("backup_2020_Sep_21/Ag_q_so", "Ag", is_quad=True)
    sp = DCCCDataProcessor("backup_2020_Sep_21/Ag_so", "Ag", is_quad=False)
    sp.load_data()
    sp.to_file("tmp.json")

    sp2 = DCCCDataProcessor.from_file("tmp.json")
    print(sp2.polar)


def test_SOCIDataProcessor():
    # sp = SOCIDataProcessor("backup_2020_Sep_21/Al_mrci", "Al", is_quad=False)
    sp = DCCIDataProcessor("backup_2020_Sep_21/O_mrci", "O", is_quad=False)
    sp.load_data()
    print(sp.find_best())


def test_AtomDataProcessor():
    adp = AtomDataProcessor("backup_2020_Sep_21", "Ag", is_quad=False)
    adp.to_file("Ag.json")
    adp = AtomDataProcessor.from_file("Ag.json")
    print(adp.analysis())


def test_Group11DataProcessor():
    # gdp = Group18DataProcessor("backup_2020_Sep_21", get_atoms_by_group(18), has_header=True)
    # gdp.to_file("g18.json")

    # gdp = GroupDataProcessor("backup_2020_Sep_21", 11, {"has_header": True})
    # gdp.to_file("g11.json")

    # gdp = Group11DataProcessor.from_file("g11.json")
    gdp = GroupDataProcessor.from_file("g11.json")
    # gdp = GroupDataProcessor.from_file("g18.json")
    # print(gdp.element)
    # print(gdp.Ag.find_best())
    gdp.summary()
