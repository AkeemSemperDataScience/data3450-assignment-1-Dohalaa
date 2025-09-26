
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


# 2 Percentage Under 30
below_30, above_equal_30 = age_splitter(df, "Age", 30)
pct_below_30 = (len(below_30) / len(df)) * 100
print("The percentage of people that are < 30 years old: " +str(round(pct_below_30, 2)) + "%")

# 3 - 1978 Earnings Comparison
# ARITHMETIC COMPARISON: we can compare the mean of each group

mean_below30 = below_30["Earnings_1978"].mean()
mean_above30 = above_equal_30["Earnings_1978"].mean()

print("Mean of the 1978 earnings of the group of people < 30 years old: " +str(round(mean_below30, 2)))
print("Mean of the 1978 earnings of the group of people >= 30 years old: " +str(round(mean_above30, 2)))

# VISUAL COMPARISON: 

# We add a column to identify age groups
df["AgeGroup"] = df["Age"].apply(lambda x: "<30" if x < 30 else ">=30")

# we create a boxplot
sns.boxplot(x = "AgeGroup", y = "Earnings_1978", data = df)

 
 
 
 
 
 
 
    
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


# 2. Effect sizes comparison

race = effectSizer(df, "Earnings_1978", "Race")
print("Race: ", round(race,3))
hisp = effectSizer(df, "Earnings_1978", "Hisp")
print("Hisp: ", round(hisp,3))
status = effectSizer(df, "Earnings_1978", "MaritalStatus")
print("Marital status: ", round(status,3))







def cohortCompare(df, cohorts, statistics=['mean', 'median', 'std', 'min', 'max']):
    """
    This function takes a dataframe and a list of cohort column names, and returns a dictionary
    where each key is a cohort name and each value is an object containing the specified statistics
    """
    # counts of the requested categorical columns
    # stats of numeric columns by cohort value

    results = {"categorical_counts": {}, "cohort_stats": {} }         

   # Numerical columns on which stats are calculated
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for cat in cohorts:
        # 1) Counts for the categorical column
        results["categorical_counts"][cat] = df[cat].value_counts(dropna=False).to_dict()

        # 2) Stats by modality (level) of this column
        results["cohort_stats"][cat] = {}
        for level, sub in df.groupby(cat, dropna=False):
            cm = CohortMetric(f"{cat}={level}")

            # Each statistic is a Series (index = numerical columns)
            if 'mean' in statistics:
                cm.setMean(sub[num_cols].mean())
            if 'median' in statistics:
                cm.setMedian(sub[num_cols].median())
            if 'std' in statistics:
                cm.setStd(sub[num_cols].std())
            if 'min' in statistics:
                cm.setMin(sub[num_cols].min())
            if 'max' in statistics:
                cm.setMax(sub[num_cols].max())

            results["cohort_stats"][cat][level] = cm

    return results


''' 2. Comparison:
At a high level, this dataset doesn't look fully representative of the U.S. population in the 
late 1970s, and even less so today. The reason is that the sample comes from a specific 
program or study group, not a random survey of the whole country. For example, the shares of people 
by education, marital status, or ethnic group don't match exactly what national statistics from the 
1970s would show.
I assessed this by looking at the distributions of demographic variables in the dataset 
(such as education level, marital status, and ethnicity) and by checking the summary statistics 
of age and earnings. When I compare these numbers with what is historically known about the U.S. 
population, there are clear differences, so the dataset can't be considered nationally 
representative.'''
  
  
# 3 - A function that prints the dictionary returned by cohortCompare in a readable way
def pretty_print(results):
    print("=== Categorical Counts ===")
    for cat in results.get("categorical_counts", {}):
        print("")
        print("Column:", cat)
        counts = results["categorical_counts"][cat]
        for k in counts:
            print(" ", k, ":", counts[k])

    print("\n=== Cohort Statistics ===")
    for cat in results.get("cohort_stats", {}):
        print("")
        print("Cohort column:", cat)
        levels = results["cohort_stats"][cat]
        for level in levels:
            cm = levels[level]
            print(cm)








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
