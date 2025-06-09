import pandas as pd

result = pd.read_csv("test_metrics.csv")

def analyze_results(result,metric_name):
    return result[metric_name].mean(), result[metric_name].std()

