import pytest
from GenomeSigInfer.utils.helpers import *


@pytest.mark.parametrize(
    "test_input, expected", [(3, 96), (5, 96 * 16), (7, 96 * 16 * 16)]
)
def test_calculate_value(test_input, expected):
    assert calculate_value(test_input) == expected


def test_calculate_value_error_low_value():
    with pytest.raises(
        ValueError, match="Input must be an uneven integer greater than or equal to 3."
    ):
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
    assert MutationalSigantures.SIZES == [
        calculate_value(i) for i in MutationalSigantures.CONTEXT_LIST[::-1]
    ]


def test_custom_sort_column_names():
    assert custom_sort_column_names("A1") == (1, "A", "")
    assert custom_sort_column_names("B10") == (10, "B", "")
    assert custom_sort_column_names("C2D") == (2, "C", "D")


def test_generate_sequence():
    assert generate_sequence(5) == [-2, 2]
    assert generate_sequence(7) == [-3, -2, 2, 3]


def test_generate_sequence_error():
    with pytest.raises(ValueError):
        generate_sequence(3)


def test_generate_numbers():
    assert generate_numbers(5) == [0, 8]
    assert generate_numbers(7) == [0, 1, 9, 10]
    assert generate_numbers(9) == [0, 1, 2, 10, 11, 12]
    assert generate_numbers(11) == [0, 1, 2, 3, 11, 12, 13, 14]


def test_generate_numbers_error():
    with pytest.raises(ValueError):
        generate_numbers(4)


def test_alphabet_list():
    assert alphabet_list(3, "SBS") == ["SBSA", "SBSB", "SBSC"]
    assert alphabet_list(5, "SBS") == ["SBSA", "SBSB", "SBSC", "SBSD", "SBSE"]


def test_create_signatures_df():
    W = np.array([[1, 2, 3], [4, 5, 6]])
    signatures_df = create_signatures_df(W, 3)
    assert isinstance(signatures_df, pd.DataFrame)
    assert signatures_df.shape == (2, 3)
    assert signatures_df.columns.tolist() == ["SBS2A", "SBS2B", "SBS2C"]


def test_calculate_value_decorator():
    @must_be_int
    def dummy_func(number):
        return number

    assert dummy_func(5) == 5

    with pytest.raises(TypeError):
        dummy_func("not_an_integer")


def test_check_supported_genome():
    check_supported_genome("GRCh37")
    check_supported_genome("GRCh38")
    with pytest.raises(error.RefGenomeNotSUpported):
        check_supported_genome("GRCh39")
