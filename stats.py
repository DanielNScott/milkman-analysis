import numpy as np
from scipy import stats


def compare_means(values_1, values_2, print_results=True, labels = ['Group 1', 'Group 2']):

    if len(values_2) == 1:
        # One-sample t-test: compare values_1 to a single point estimate (mean of values_2)
        t_stat, p_value = stats.ttest_1samp(values_1, values_2[0])
    else:
        # Perform independent t-test (two-sided by default)
        t_stat, p_value = stats.ttest_ind(values_1, values_2)

    # Calculate Cohen's d
    d_prime = cohen_d(values_1, values_2)

    # Output the results
    if print_results:
        print(f"Comparing {labels[0]} and {labels[1]}:")
        print(f"T-statistic: {t_stat:.2e}")
        print(f"P-value: {p_value:.2e}")
        print(f"Cohen's d: {d_prime:.2e}")
        print("\n")

    return p_value, d_prime


# Calculate Cohen's d for effect size
def cohen_d(x, y):
    # Calculate the means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate the pooled standard deviation
    if len(y) == 1:
        pooled_std  = np.std(x, ddof=1)
    else:
        pooled_std = np.sqrt(((len(x) - 1) * np.std(x, ddof=1) ** 2 + (len(y) - 1) * np.std(y, ddof=1) ** 2) / (len(x) + len(y) - 2))        

    # Compute Cohen's d
    return (mean_x - mean_y) / pooled_std