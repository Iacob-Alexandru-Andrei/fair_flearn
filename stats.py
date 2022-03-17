import statistics as stats
import pandas as pd

file_name = "log_synthetic/gini_ffedavg_samp2_run1_q0_2000_validation.csv"

df = pd.read_csv(file_name)
print("Describe:")
print(df.describe())


print("Worst 10%")
print(df[df.columns[0]].nsmallest(int(len(df)/10)).mean())


print("Best 10%")
print(df[df.columns[0]].nlargest(int(len(df)/10)).mean())

print("Var:")
print(stats.variance(df[df.columns[0]]))

