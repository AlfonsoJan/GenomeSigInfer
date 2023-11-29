import pytest
from GenomeSigInfer.utils.helpers import *

@pytest.mark.parametrize("test_input, expected", [(3, 96), (5, 96*16), (7, 96*16*16)])
def test_calculate_value(test_input, expected):
    assert calculate_value(test_input) == expected

def test_calculate_value_error_low_value():
    with pytest.raises(ValueError, match="Input must be an uneven integer greater than or equal to 3."):
        calculate_value(2)

def test_calculate_value_error_not_int():
    with pytest.raises(TypeError, match="Input must be an integer."):
        calculate_value("not_an_integer")

def test_mutational_signatures():
    # Test MutationalSigantures class attributes
    assert MutationalSigantures.REF_GENOMES == ["GRCh37", "GRCh38"]
    assert MutationalSigantures.MAX_CONTEXT == 7
    assert MutationalSigantures.SORT_REGEX == {
        13: r"(\w\w\w\w\w\w\[.*\]\w\w\w\w\w\w)",
        11: r"(\w\w\w\w\w\[.*\]\w\w\w\w\w)",
        9: r"(\w\w\w\w\[.*\]\w\w\w\w)",
        7: r"(\w\w\w\[.*\]\w\w\w)",
        5: r"(\w\w\[.*\]\w\w)",
        3: r"(\w\[.*\]\w)",
    }

    # Test MutationalSigantures.SIZES based on calculate_value function
    assert MutationalSigantures.SIZES == [calculate_value(i) for i in MutationalSigantures.CONTEXT_LIST[::-1]]
