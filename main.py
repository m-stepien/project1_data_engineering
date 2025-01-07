from dataload import load_from_file
from vizualization import show_histogram, show_scatter_plot, show_heatmap
import pandas as pd
import numpy as np
df = load_from_file("resource/data.csv", {})
print(df)
numeric_values = df.select_dtypes(include=[np.number])
selected_columns = numeric_values.iloc[:, :25]

# show_histogram(numeric_values)
# show_scatter_plot(df["Age"], df["Income"], x_name="Age", y_name="Income")
corr_matrix = selected_columns.corr()
show_heatmap(corr_matrix)