from dataload import load_from_file
from vizualization import show_histogram, show_scatter_plot, show_heatmap, show_missing_data, data_distribution
import pandas as pd
import numpy as np
df = load_from_file("resource/data.csv", {})
print(df)
numeric_values = df.select_dtypes(include=[np.number])
selected_columns = numeric_values.iloc[:, :25]


# show_histogram(numeric_values)
# show_scatter_plot(df["Age"], df["Income"], x_name="Age", y_name="Income")
# corr_matrix = selected_columns.corr()
# show_heatmap(corr_matrix)

# np.random.seed(42)
# data = {
#     'A': np.random.choice([1, 2, np.nan], size=100, p=[0.6, 0.3, 0.1]),
#     'B': np.random.choice([5, np.nan], size=100, p=[0.8, 0.2]),
#     'C': np.random.choice([np.nan, 10, 20], size=100, p=[0.15, 0.5, 0.35]),
#     'D': np.random.choice([3, 4, np.nan], size=100, p=[0.7, 0.2, 0.1]),
#     'E': np.random.choice([np.nan, 7], size=100, p=[0.05, 0.95]),
# }
#
# df_with_nan = pd.DataFrame(data)
# df_with_nan['E'] = np.nan
# show_missing_data(df_with_nan)

# np.random.seed(42)
# data = {
#     'Kategoria': np.random.choice(['A', 'B', 'C', 'D'], size=100),
#     'Wartości': np.random.randint(1, 100, size=100)
# }
#
# df_categorical = pd.DataFrame(data)
# df_categorical['Kategoria'] = df_categorical['Kategoria'].replace('A', 'E')
# # data_distribution(df_categorical["Kategoria"])
# print(df_categorical["Wartości"])
# data_distribution(df_categorical["Wartości"], "Wartości")
