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
                print(f"{idx + 1}. {key}")
            print("0. Kontynuuj")
            choice = input()
            if choice != "0":
                value = input("Podaj wartość wybranego parametru:\t")
                personal_parameters[list(atributes_keys)[int(choice) - 1]] = value
            print(personal_parameters)
    return personal_parameters


def exploration_menu():
    print(
        "1. Histogram\n2. Wykres punktowy \n3. Korelacja między zmiennymi\n4.Analiza braków danych\n5. Rozkład zmiennych\n"
        "6. Podsumowanie statystyczne")
    choice = input()
    return choice


def vizualization_menu():
    print("1. Histogram\n2. Scatter plot\n3. Correlations between variables\n0. Back")


def choose_column(columns_name):
    for idx, column in enumerate(columns_name):
        print(f"{idx + 1}. {column}")
    choice = None
    while choice is None:
        try:
            choice = int(input())
            if 1 <= choice <= len(columns_name):
                return columns_name[choice - 1]
            else:
                print("You must choose from index of columns")
                choice = None
        except Exception as e:
            print("Choice must by a number")



def main():
    filename = file_name_screen()
    atributes = additional_parameters(filename)
    df = load_from_file(filename, atributes)
    print(df.head())
    main_choice = None
    while main_choice != "0":
        print(
            "1. Vizualization\n2. Statistic summary\n3. Data preparation\n4. Model selection\n5. Budowanie modelu\n6. Model evaluation\n"
            "6. Export results\n0.Exit")
        main_choice = input()
        if main_choice == "1":
            print("1. Histogram\n2. Scatter plot\n3. Correlations between variables\n4. Missing data analysis\n"
                  "5. Distribution of variables\n 0. Back")
            visualization_choice = None
            while visualization_choice != "0":
                visualization_choice = input()
                if visualization_choice == "1":
                    print("Select column")
                    for index, column in enumerate(df.columns):
                        print(f"{index + 1}. {column}")
                    print(f"{len(df.columns) + 1}. SELECT ALL")
                    print("0. Back")
                    column_selection = None
                    while column_selection != "0":
                        try:
                            column_selection = int(input())
                        except Exception as e:
                            print("choice must be a number")
                            continue
                        if column_selection == len(df.columns) + 1:
                            pass
                        elif 0 < column_selection <= len(df.columns):
                            column_name = df.columns[column_selection - 1]
                            show_histogram(df[column_name], column_name)
                        elif column_selection == "0":
                            continue
                elif visualization_choice == "2":
                    numerical_columns = df.select_dtypes(exclude=['object', 'category', 'string']).columns.tolist()
                    column_name_1 = choose_column(numerical_columns)
                    numerical_columns.remove(column_name_1)
                    column_name_2 = choose_column(numerical_columns)
                    show_scatter_plot(df[column_name_1], df[column_name_2], column_name_1, column_name_2)
                elif visualization_choice == "3":
                    corr = calc_corr_for_all(df.select_dtypes(exclude=['object', 'category', 'string']))
                    show_heatmap(corr)
                elif visualization_choice == "4":
                    show_missing_data(df)
                elif visualization_choice == "5":
                    print("Select column")
                    for index, column in enumerate(df.columns):
                        print(f"{index + 1}. {column}")
                    print("0. Back")
                    column_selection = None
                    while column_selection != "0":
                        try:
                            column_selection = int(input())
                        except Exception as e:
                            print("choice must be a number")
                            continue
                        if 0 < column_selection <= len(df.columns):
                            column_name = df.columns[column_selection - 1]
                            data_distribution(df[column_name], column_name)
                        elif column_selection == "0":
                            continue

                    data_distribution(df["Income"])
                elif visualization_choice == "0":
                    continue
                else:
                    print("No such option")
        elif main_choice == "2":
            pass
        elif main_choice == "3":
            pass
        elif main_choice == "4":
            pass
        elif main_choice == "5":
            pass
        elif main_choice == "6":
            pass
        elif main_choice == "0":
            continue
        else:
            print("Sorry but there is no such option")

    vizualization_menu()


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
