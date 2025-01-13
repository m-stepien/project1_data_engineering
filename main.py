from dataload import load_from_file, get_atributes_list_for_file_extension
from vizualization import show_histogram, show_scatter_plot, show_heatmap, show_missing_data, data_distribution
from statistical_summary import calc_corr_for_all, split_df_to_categorical_and_numerical
import pandas as pd
import numpy as np


def file_name_screen():
    filename = input("Podaj nazwę pliku:\t")
    return filename


def additional_parameters(fname):
    personal_parameters = {}
    atributes_keys = get_atributes_list_for_file_extension(fname.split(".")[-1])
    yn = input("Czy chcesz dodać niestandardowe parametry do otwarcia pliku (y/n)\t")
    if yn == 'y':
        choice = None
        while choice != "0":
            for idx, key in enumerate(atributes_keys):
                print(f"{idx+1}. {key}")
            print("0. Kontynuuj")
            choice = input()
            if choice != "0":
                value = input("Podaj wartość wybranego parametru:\t")
                personal_parameters[list(atributes_keys)[int(choice)-1]] = value
            print(personal_parameters)
    return personal_parameters


def exploration_menu():
    print("1. Histogram\n2. Wykres punktowy \n3. Korelacja między zmiennymi\n4.Analiza braków danych\n5. Rozkład zmiennych\n"
          "6. Podsumowanie statystyczne")
    choice = input()
    return choice

def data_preparation_menu():



def main():
    filename = file_name_screen()
    atributes = additional_parameters(filename)
    # load_from_file(filename, atributes)

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

if __name__ == '__main__':
    main()
