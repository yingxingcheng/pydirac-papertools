import importlib.resources as pkg_resources
import tempfile

import pytest

import tests.data  # Ensure your tests/data is a Python package (i.e., it contains an __init__.py file)
from pydirac_papertools.processor import (
    AtomDataProcessor,
    DCCCDataProcessor,
    DCCIDataProcessor,
    GroupDataProcessor,
    NRCCDataProcessor,
    SRCCDataProcessor,
)


@pytest.mark.parametrize("data_path, element", [("Al_nr", "Al")])
def test_NRCCDataProcessor(data_path, element):
    with pkg_resources.path(tests.data, data_path) as data_file:
        np = NRCCDataProcessor(str(data_file), element)
        print(np.find_best())


@pytest.mark.parametrize(
    "data_path, element, is_quad",
    [("Al", "Al", False)],
)
def test_SRCCDataProcessor(data_path, element, is_quad):
    with pkg_resources.path(tests.data, data_path) as data_file:
        sp_d = SRCCDataProcessor(str(data_file), element, is_quad=is_quad)
        print(sp_d.as_dict())


@pytest.mark.parametrize("data_path, element, is_quad", [("Al_so", "Al", False)])
def test_SOCCDataProcessor(data_path, element, is_quad):
    with pkg_resources.path(tests.data, data_path) as data_file:
        sp = DCCCDataProcessor(str(data_file), element, is_quad=is_quad)
        sp.load_data()
        print(sp.polar)


@pytest.mark.parametrize("data_path, element, is_quad", [("O_mrci", "O", False)])
def test_SOCIDataProcessor(data_path, element, is_quad):
    with pkg_resources.path(tests.data, data_path) as data_file:
        sp = DCCIDataProcessor(str(data_file), element, is_quad=is_quad)
        sp.load_data()
        print(sp.find_best())


@pytest.mark.parametrize("data_path, element, is_quad", [(".", "Al", False)])
def test_AtomDataProcessor(data_path, element, is_quad):
    with pkg_resources.path(tests.data, data_path) as data_file:
        adp = AtomDataProcessor(str(data_file), element, is_quad=is_quad)
        print(adp.analysis())


@pytest.mark.parametrize("file_path", [("group-1.json")])
def test_GroupDataProcessor(file_path):
    with pkg_resources.path(tests.data, file_path) as data_file:
        gdp = GroupDataProcessor.from_file(str(data_file))
        print(gdp)
        gdp.summary()


# New tests with temporary directory for caching
@pytest.mark.parametrize("data_path, element, is_quad", [("Al_so", "Al", False)])
def test_SOCCDataProcessor_to_file(data_path, element, is_quad):
    with pkg_resources.path(tests.data, data_path) as data_file:
        sp = DCCCDataProcessor(str(data_file), element, is_quad=is_quad)
        sp.load_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = f"{temp_dir}/tmp.json"
            sp.to_file(temp_file)
            sp2 = DCCCDataProcessor.from_file(temp_file)
            print(sp2.polar)


@pytest.mark.parametrize("data_path, element, is_quad", [(".", "Al", False)])
def test_AtomDataProcessor_to_file(data_path, element, is_quad):
    with pkg_resources.path(tests.data, data_path) as data_file:
        adp = AtomDataProcessor(str(data_file), element, is_quad=is_quad)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = f"{temp_dir}/Al.json"
            adp.to_file(temp_file)
            adp = AtomDataProcessor.from_file(temp_file)
            print(adp.analysis())
