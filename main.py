from dataload import load_from_file, get_atributes_list_for_file_extension
from vizualization import show_histogram, show_scatter_plot, show_heatmap, show_missing_data, data_distribution
from statistical_summary import calc_statistic_for_all_columns, calc_corr_for_all, split_df_to_categorical_and_numerical
from data_preparation import removing_missing_data, imputation_missing_data, data_standardization, data_normalization, \
    one_hot_encoding, label_encoding, split_to_test_training
from model import train_decision_tree_classifier, train_knn
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


def select_test_data_percentage():
    while True:
        try:
            percent = float(input("Select percent of data that will test model (0-100): "))
            if 0 < percent < 100:
                return percent / 100
            else:
                print("Number must be between 0 and 100")
        except Exception:
            print("Input must be a number")


def get_criterion():
    while True:
        print("Select criterion for decision tree\n1. Gini\n2. Entropy")
        choice = input()
        if choice == "1":
            return "gini"
        elif choice == "2":
            return "entropy"
        else:
            print("You must select 1 or 2")


def get_max_depth():
    while True:
        choice = input("\nEnter the max depth (positive integer or 0 for no limits): ")
        if choice.lower() == "0":
            return None
        try:
            max_depth = int(choice)
            if max_depth > 0:
                return max_depth
            else:
                print("Max depth can't be negative")
        except ValueError:
            print("You must give number as input")


def get_n_neighbors():
    while True:
        try:
            n_neighbors = int(input("\nEnter the number of neighbors for KNN: ").strip())
            if n_neighbors > 0:
                return n_neighbors
            else:
                print("Number of neighbors must be a positive integer")
        except ValueError:
            print("Input must be a number")


def get_knn_metric():
    metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine', 'hamming']
    print("Select metrics for KNN:")
    for i, metric in enumerate(metrics):
        print(f"{i+1}. {metric}")
    while True:
        choice = input("Choose a metric by entering the number: ").strip()
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(metrics):
                return metrics[choice_index]
            else:
                print("Invalid choice. Please enter a valid number from the list")
        except ValueError:
            print("Invalid input")


def main():
    data_splitted = None
    filename = file_name_screen()
    atributes = additional_parameters(filename)
    df = load_from_file(filename, atributes)
    print(df.head())
    main_choice = None
    while main_choice != "0":
        print(
            "1. Vizualization\n2. Statistic summary\n3. Data preparation\n4. Model selection\n5. Building model\n6. Model evaluation\n"
            "6. Export results\n0.Exit")
        main_choice = input()
        if main_choice == "1":
            visualization_choice = None
            while visualization_choice != "0":
                print("1. Histogram\n2. Scatter plot\n3. Correlations between variables\n4. Missing data analysis\n"
                      "5. Distribution of variables\n 0. Back")
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
            statistic_summary_choice = None
            while statistic_summary_choice != "0":
                print("1. Summary\n2. Corelation\n0. Back")
                statistic_summary_choice = input()
                if statistic_summary_choice == "1":
                    results = calc_statistic_for_all_columns(df)
                    print("Summary for numerical values")
                    print(results[0])
                    print("Summary for categorical values")
                    print(results[1])
                elif statistic_summary_choice == "2":
                    correlation = calc_corr_for_all(df)
                    print(correlation.to_string())
        elif main_choice == "3":
            data_preparation_choice = None
            while data_preparation_choice != "0":
                print("1. Data cleaning\n2. Coding categorical variable\n3. Splitting to training and testing data"
                      "\n0. Back")
                data_preparation_choice = input()
                if data_preparation_choice == "1":
                    data_cleaning_choice = None
                    while data_cleaning_choice != "0":
                        print("1. Deleting missing data\n2. Imputation of missing data\n3. Data standardization\n"
                              "4. Data normalization\n0. Back")
                        data_cleaning_choice = input()
                        if data_cleaning_choice == "1":
                            removing_missing_data(df)
                        elif data_cleaning_choice == "2":
                            df = imputation_missing_data(df)
                        elif data_cleaning_choice == "3":
                            df = data_standardization(df)
                        elif data_cleaning_choice == "4":
                            df = data_normalization(df)
                        elif data_cleaning_choice == "0":
                            continue
                        else:
                            print("No such option")
                elif data_preparation_choice == "2":
                    categorical_encoding_choice = None
                    while categorical_encoding_choice != "0":
                        print("1. One-Hot Encoding\n2. Label Encoding\n0. Back")
                        categorical_encoding_choice = input()
                        if categorical_encoding_choice == "1":
                            df = one_hot_encoding(df)
                            print(df.head().to_string())
                        elif categorical_encoding_choice == "2":
                            df = label_encoding(df)
                            print(df.head().to_string())
                        elif categorical_encoding_choice == "0":
                            continue
                        else:
                            print("No such option")
                elif data_preparation_choice == "3":
                    columns_names = df.select_dtypes(include=['number']).columns.to_list()
                    x_column_names = []
                    print("Select label")
                    y_label = choose_column(columns_names)
                    x_column_choice_idx = None
                    columns_names.remove(y_label)
                    print("Select features")
                    while x_column_choice_idx != 0:
                        for idx, column_name in enumerate(columns_names):
                            print(f"{idx + 1}. {column_name}")
                        print("0. Continue")
                        try:
                            x_column_choice_idx = int(input())
                        except Exception:
                            print("Choice must be a number")
                            continue
                        if 1 <= x_column_choice_idx <= len(columns_names):
                            selected_column = columns_names.pop(x_column_choice_idx - 1)
                            x_column_names.append(selected_column)
                            print(x_column_names)
                        elif x_column_choice_idx == 0:
                            if len(x_column_names) == 0:
                                print("You must select at lest one column as feature")
                                x_column_choice_idx = None
                    test_size = select_test_data_percentage()
                    data_splitted = split_to_test_training(df[x_column_names], df[y_label], test_size)
                    print(data_splitted)
                elif data_preparation_choice == "0":
                    continue
                else:
                    print("No such option")
        elif main_choice == "4":
            if data_splitted is None:
                print("First you must split data to training and testing")
            else:
                model_category_choice = None
                while model_category_choice != "0":
                    print("1. Classification models\n2. Regression model\n0. Back")
                    model_category_choice = input()
                    if model_category_choice == "1":
                        categorical_model_choice = None
                        while categorical_model_choice != "0":
                            print(
                                "1. Decision trees\n2. KNN algorithm\n3. Random Forest\n4. SVM\n5. Logistic regression"
                                "\n0. Back")
                            categorical_model_choice = input()
                            if categorical_model_choice == "1":
                                criterion = get_criterion()
                                max_depth = get_max_depth()
                                try:
                                    model = train_decision_tree_classifier(data_splitted[0], data_splitted[2], criterion, max_depth)
                                    print(model)
                                except ValueError:
                                    print("You select continuous type as label you must use one of regresion model instead")
                                    categorical_model_choice = "0"
                            elif categorical_model_choice == "2":
                                neighbors = get_n_neighbors()
                                metric = get_knn_metric()
                                model = train_knn(data_splitted[0], data_splitted[2], neighbors, metric)
                                print(model)
                            elif categorical_model_choice == "3":
                                pass
                            elif categorical_model_choice == "4":
                                pass
                            elif categorical_model_choice == "5":
                                pass
                            elif categorical_model_choice == "0":
                                continue
                    elif model_category_choice == "2":
                        print("1. Linear regression\n2. Regression trees\n3. Random Forest")
        elif main_choice == "5":
            pass
        elif main_choice == "6":
            pass
        elif main_choice == "0":
            continue
        else:
            print("Sorry but there is no such option")


if __name__ == '__main__':
    main()
