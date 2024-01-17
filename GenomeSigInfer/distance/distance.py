#!/usr/bin/env python3
"""
Distance Calculation Module

This module provides functions for calculating optimal column assignments between two DataFrames based on the linear sum assignment and calculating Jensen-Shannon distance for each pair of columns.

Functions:
* calculate_distances(control_df: pd.DataFrame, df_not_compressed: pd.DataFrame) -> tuple[pd.DataFrame]: Calculate Jensen Shannon Distance and Cosine Similarity between two matrices
* calculate_similarity(similarity_function: callable, optimal_columns: dict, control_df: np.ndarray, df_compare: np.ndarray) -> pd.DataFrame: Calculate similarity or distance between two matrices using a specified function. Either cosine or jensen shannon distance
* create_decompose_df(columns: dict, df: pd.DataFrame) -> pd.DataFrame: Create decompose dataframe
* get_optimal_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> dict: Get the optimal column assignments between two DataFrames based on the linear sum assignment.
* set_optimal_columns(control_df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame: Set optimal column assignments for the control DataFrame based on the linear sum assignment.
* get_jensen_shannon_distance(optimal_columns: dict, df1: pd.DataFrame,df2: pd.DataFrame) -> pd.DataFrame: Calculate Jensen-Shannon distance for each pair of columns based on optimal column assignments.
        
Author: J.A. Busker
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from ..matrix import matrix_operations
from . import cosine


def calculate_distances(
	control_df: pd.DataFrame, df_not_compressed: pd.DataFrame
) -> tuple[pd.DataFrame]:
	"""
	Calculate Jensen Shannon Distance and Cosine Similarity between two matrices.

	Args:
	    control_df (pd.DataFrame): The control matrix.
	    df_not_compressed (pd.DataFrame): The matrix to compare.

	Returns:
	    Tuple: DataFrames of Jensen Shannon Distance and Cosine Similarity.
	"""
	# Compress df_not_compressed to a 96-well format
	df_compare = matrix_operations.compress_to_96(df_not_compressed)
	df_compare = df_compare.iloc[:, 1:]
	# Get optimal column assignments between control_df and df_compare
	optimal_columns = get_optimal_columns(control_df, df_compare)
	# Create a decomposed DataFrame based on optimal column assignments
	decomposed_df = create_decompose_df(optimal_columns, df_not_compressed)
	# Calculate Jensen Shannon Distance and Cosine Similarity
	result_jens_df = calculate_similarity(
		get_jensen_shannon_distance, optimal_columns, control_df, df_compare
	)
	result_cosine_df = calculate_similarity(
		cosine.cosine_nmf_w, optimal_columns, control_df, df_compare
	)
	return result_jens_df, result_cosine_df, decomposed_df


def calculate_similarity(
	similarity_function: callable,
	optimal_columns: dict,
	control_df: np.ndarray,
	df_compare: np.ndarray,
) -> pd.DataFrame:
	"""
	Calculate similarity or distance between two matrices using a specified function.
	Either cosine or jensen shannon distance

	Args:
	    similarity_function (Callable): The similarity or distance function.
	    optimal_columns (dict): Optimal columns for comparison.
	    control_df (np.ndarray): The control matrix.
	    df_compare (np.ndarray): The matrix to compare.

	Returns:
	    DataFrame: Result values.
	"""
	# Call the specified similarity function with optimal columns, control_df, and df_compare
	result = similarity_function(
		optimal_columns=optimal_columns, df1=control_df, df2=df_compare
	)
	# Create a DataFrame with the result values
	result_df = pd.concat([pd.DataFrame(), pd.DataFrame([result])], ignore_index=True)
	return result_df


def create_decompose_df(columns: dict, df: pd.DataFrame) -> pd.DataFrame:
	"""
	Create decompose dataframe.

	Args:
	    columns (dict): A dictionary where keys are column names from df1,
	        and values are corresponding column names from df2.
	    df (pd.DataFrame): The matrix for the new columns.

	Returns:
	    pd.DataFrame: With the updated columns
	"""
	# Create a copy of the original DataFrame
	df_copy = df.copy()
	# Drop the 'sort_key' column if it exists
	df_copy = df_copy.drop("sort_key", axis=1)
	# Rename columns based on the provided mapping in the 'columns' dictionary
	df_copy.rename(columns={value: key for key, value in columns.items()}, inplace=True)
	return df_copy


def get_optimal_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
	"""
	Get the optimal column assignments between two DataFrames based on the linear sum assignment.

	Args:
	    df1 (pd.DataFrame): The first DataFrame.
	    df2 (pd.DataFrame): The second DataFrame.

	Returns:
	    dict: A dictionary where keys are column names from df1,
	        and values are corresponding column names from df2.
	"""
	# Initialize a cost matrix to store the pairwise differences between columns
	cost_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
	# Fill the cost matrix with squared differences between each pair of columns
	for col1 in df1.columns:
		for col2 in df2.columns:
			cost_matrix.loc[col1, col2] = ((df1[col1] - df2[col2]) ** 2).sum()
	# Convert the cost matrix to a NumPy array for linear sum assignment
	cost_matrix_array = cost_matrix.values.astype(float)
	# Use linear sum assignment to find the optimal column assignments
	row_indices, col_indices = linear_sum_assignment(cost_matrix_array)
	# The col_indices represent the optimal column assignments
	optimal_column_assignments = dict(
		zip(df1.columns[row_indices], df2.columns[col_indices])
	)
	return optimal_column_assignments


def set_optimal_columns(control_df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
	"""
	Set optimal column assignments for the control DataFrame based on the linear sum assignment.

	Args:
	    control_df (pd.DataFrame): The control DataFrame.
	    df2 (pd.DataFrame): The second DataFrame.

	Returns:
	    pd.DataFrame: The control DataFrame with optimal column assignments.
	"""
	# Get optimal column assignments between control_df and df2
	# Set the column names in the control DataFrame based on the optimal assignments
	control_df.columns = ["MutationType"] + list(
		get_optimal_columns(control_df.iloc[:, 1:], df2.iloc[:, 1:]).values()
	)
	return control_df


def get_jensen_shannon_distance(
	optimal_columns: dict, df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
	"""
	Calculate Jensen-Shannon distance for each pair of columns based on optimal column assignments.

	Parameters:
	    optimal_columns (dict): Dictionary containing optimal column assignments between df1 and df2.
	    df1 (pd.DataFrame): The first DataFrame.
	    df2 (pd.DataFrame): The second DataFrame.

	Returns:
	    pd.DataFrame: A DataFrame containing Jensen-Shannon distances for each pair of columns.
	"""
	# Initialize an empty dictionary to store Jensen-Shannon distances
	row = {}
	# Iterate over the optimal column assignments
	for col1, col2 in optimal_columns.items():
		# Calculate Jensen-Shannon distance for the current pair of columns
		distance = jensenshannon(df1[col1], df2[col2])
		# Add the distance to the dictionary
		row[col1] = distance
	return row
