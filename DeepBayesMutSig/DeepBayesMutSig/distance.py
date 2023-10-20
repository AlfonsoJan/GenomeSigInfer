
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment

def get_optimal_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """
    Get the optimal column assignments between two DataFrames based on the linear sum assignment.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        dict: A dictionary where keys are column names from df1, and values are corresponding column names from df2.
    """
    cost_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for col1 in df1.columns:
        for col2 in df2.columns:
            cost_matrix.loc[col1, col2] = ((df1[col1] - df2[col2]) ** 2).sum()
    cost_matrix_array = cost_matrix.values.astype(float)
    row_indices, col_indices = linear_sum_assignment(cost_matrix_array)

    # The col_indices represent the optimal column assignments
    optimal_column_assignments = dict(
        zip(df1.columns[row_indices], df2.columns[col_indices])
    )
    return optimal_column_assignments

def get_jensen_shannon_distance(optimal_columns: dict, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Jensen-Shannon distance for each pair of columns based on optimal column assignments.

    Parameters:
        optimal_columns (dict): Dictionary containing optimal column assignments between df1 and df2.
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing Jensen-Shannon distances for each pair of columns.
    """
    row = {}
    for col1, col2 in optimal_columns.items():
        distance = jensenshannon(df1[col1], df2[col2])
        row[col1] = distance
    return row
