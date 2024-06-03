import tempfile

import pytest

from pydirac_papertools.processor import (
    AtomDataProcessor,
    DCCCDataProcessor,
    DCCIDataProcessor,
    GroupDataProcessor,
    NRCCDataProcessor,
    SRCCDataProcessor,
)


@pytest.mark.parametrize("data_path, element", [("data/Al_nr", "Al")])
def test_NRCCDataProcessor(data_path, element):
    np = NRCCDataProcessor(data_path, element)
    print(np.find_best())


@pytest.mark.parametrize(
    "data_path, element, is_quad",
    [("data/Al", "Al", False)],
)
def test_SRCCDataProcessor(data_path, element, is_quad):
    sp_d = SRCCDataProcessor(data_path, element, is_quad=is_quad)
    print(sp_d.as_dict())


@pytest.mark.parametrize("data_path, element, is_quad", [("data/Al_so", "Al", False)])
def test_SOCCDataProcessor(data_path, element, is_quad):
    sp = DCCCDataProcessor(data_path, element, is_quad=is_quad)
    sp.load_data()
    print(sp.polar)


@pytest.mark.parametrize("data_path, element, is_quad", [("data/O_mrci", "O", False)])
def test_SOCIDataProcessor(data_path, element, is_quad):
    sp = DCCIDataProcessor(data_path, element, is_quad=is_quad)
    sp.load_data()
    print(sp.find_best())


@pytest.mark.parametrize("data_path, element, is_quad", [("data", "Al", False)])
def test_AtomDataProcessor(data_path, element, is_quad):
    adp = AtomDataProcessor(data_path, element, is_quad=is_quad)
    print(adp.analysis())


@pytest.mark.parametrize("file_path", [("data/group-1.json")])
def test_GroupDataProcessor(file_path):
    gdp = GroupDataProcessor.from_file(file_path)
    print(gdp)
    gdp.summary()


# New tests with temporary directory for caching
@pytest.mark.parametrize("data_path, element, is_quad", [("data/Al_so", "Al", False)])
def test_SOCCDataProcessor_to_file(data_path, element, is_quad):
    sp = DCCCDataProcessor(data_path, element, is_quad=is_quad)
    sp.load_data()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = f"{temp_dir}/tmp.json"
        sp.to_file(temp_file)
        sp2 = DCCCDataProcessor.from_file(temp_file)
        print(sp2.polar)


@pytest.mark.parametrize("data_path, element, is_quad", [("data", "Al", False)])
def test_AtomDataProcessor_to_file(data_path, element, is_quad):
    adp = AtomDataProcessor(data_path, element, is_quad=is_quad)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = f"{temp_dir}/Al.json"
        adp.to_file(temp_file)
        adp = AtomDataProcessor.from_file(temp_file)
        print(adp.analysis())
