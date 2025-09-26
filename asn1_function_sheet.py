
import pandas as pd
import numpy as np
import math

def age_splitter(df, col_name, age_threshold):
    """
    Splits the dataframe into two dataframes based on an age threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    col_name (str): The name of the column containing age values.
    age_threshold (int): The age threshold for splitting.

    Returns:
    tuple: A tuple containing two dataframes:
        - df_below: DataFrame with rows where age is below the threshold.
        - df_above_equal: DataFrame with rows where age is above or equal to the threshold.
    """
    below = df[ df[col_name] < age_threshold ]     
    above_equal = df[ df[col_name] >= age_threshold ]
    return below, above_equal


 
 
 
 
 
 
 
    
def effectSizer(df, num_col, cat_col):
    """
    Calculates the effect sizes of binary categorical classes on a numerical value.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    num_col (str): The name of the numerical column.
    cat_col (str): The name of the binary categorical column.

    Returns:
    float: Cohen's d effect size between the two groups defined by the categorical column.
    Raises:
    ValueError: If the categorical column does not have exactly two unique values.
    """
    values = df[cat_col].unique()
    group1 = df[df[cat_col] == values[0]][num_col]
    group2 = df[df[cat_col] == values[1]][num_col]

    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    v1, v2 = group1.var(), group2.var()
    diff = m1 - m2
    
    pooled_var = (n1 * v1 + n2 * v2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d








def cohortCompare(df, cohorts, statistics=['mean', 'median', 'std', 'min', 'max']):
    """
    This function takes a dataframe and a list of cohort column names, and returns a dictionary
    where each key is a cohort name and each value is an object containing the specified statistics
    """
    result = {}

    # Numerical columns 
    numeric_cols = df.select_dtypes(include="number").columns.difference(cohorts)

    # numerical
    for col in numeric_cols:
        metric = CohortMetric("numeric")
        metric.set_values(df[col].dropna(), statistics)
        result[col] = metric.to_dict()

    # categorical
    for col in cohorts:
        metric = CohortMetric("categorical")
        metric.set_values(df[col], statistics)
        result[col] = metric.to_dict()

    return result








class CohortMetric():
    # don't change this
    def __init__(self, cohort_name):
        self.cohort_name = cohort_name
        self.statistics = {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }
    def setMean(self, new_mean):
        self.statistics["mean"] = new_mean
    def setMedian(self, new_median):
        self.statistics["median"] = new_median
    def setStd(self, new_std):
        self.statistics["std"] = new_std
    def setMin(self, new_min):
        self.statistics["min"] = new_min
    def setMax(self, new_max):
        self.statistics["max"] = new_max

    def compare_to(self, other):
        for stat in self.statistics:
            if not self.statistics[stat].equals(other.statistics[stat]):
                return False
        return True
    def __str__(self):
        output_string = f"\nCohort: {self.cohort_name}\n"
        for stat, value in self.statistics.items():
            output_string += f"\t{stat}:\n{value}\n"
            output_string += "\n"
        return output_string
