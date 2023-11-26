
import pandas as pd

def assert_dataframes_equal(file1: pd.DataFrame, file2: pd.DataFrame) -> None:
    """
    Function that asserts if 2 dataframes are equal
    """
    df = pd.read_csv(file1, sep=",")
    df_other = pd.read_csv(file2, sep=",")
    
    # Sort columns
    df = df.sort_index(axis=1)
    df_other = df_other.sort_index(axis=1)
    
    assert df.equals(df_other)

def test_generate_sbs_matrix1():
    """
    """
    assert_dataframes_equal("src/tests/data/sbs.96.txt", "src/tests/data/WGS_Other.96.csv")

def test_generate_sbs_matrix2():
    """
    """
    assert_dataframes_equal("src/tests/data/sbs.1536.txt", "src/tests/data/WGS_Other.1536.csv")
