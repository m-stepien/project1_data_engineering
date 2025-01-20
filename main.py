from dataload import load_from_file, get_atributes_list_for_file_extension
from vizualization import *
from statistical_summary import *
from data_preparation import removing_missing_data, imputation_missing_data, data_standardization, data_normalization, \
    one_hot_encoding, label_encoding, split_to_test_training
from model import *
import pandas as pd
import numpy as np

MODELS = {
    "decision tree": train_decision_tree_classifier,
    "knn": train_knn,
    "random forest": train_random_forest,
    "svm": train_svc,
    "logistic regression": train_logistic_regression,
    "linear regression": train_linear_regression,
    "regression tree": train_decision_tree_regressor,
    "random forest (regression)": train_random_forest_regressor
}


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


def is_label_categorical(label):
    if label.dtype == "object" or label.dtype.name == "category":
        return True
    if np.issubdtype(label.dtype, np.floating):
        return False
    else:
        return True


def main():
    data_splitted = None
    selected_model = None
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
                if is_label_categorical(data_splitted[2]):
                    categorical_model_choice = None
                    while categorical_model_choice != "0":
                        print(
                            "1. Decision tree\n2. KNN algorithm\n3. Random Forest\n4. SVM\n5. Logistic regression"
                            "\n0. Back")
                        categorical_model_choice = input()
                        if categorical_model_choice == "1":
                            selected_model = "decision tree"
                        elif categorical_model_choice == "2":
                            selected_model = "knn"
                        elif categorical_model_choice == "3":
                            selected_model = "random forest"
                        elif categorical_model_choice == "4":
                            selected_model = "svm"
                        elif categorical_model_choice == "5":
                            selected_model = "logistic regression"
                        print(f"Selected model {selected_model}")
                else:
                    regression_model_choice = None
                    while regression_model_choice != "0":
                        print("1. Linear regression\n2. Regression tree\n3. Random Forest\n0. Back")
                        regression_model_choice = input()
                        if regression_model_choice == "1":
                            selected_model = "linear regression"
                        elif regression_model_choice == "2":
                            selected_model = "regression tree"
                        elif regression_model_choice == "3":
                            selected_model = "random forest (regression)"
                        print(f"Selected model {selected_model}")
        elif main_choice == "5":
            if selected_model is not None:
                model = MODELS[selected_model](data_splitted[0], data_splitted[2])
                y_pred = model.predict(data_splitted[1])
                if is_label_categorical(data_splitted[2]):
                    y_pred_prob = model.predict_proba(data_splitted[1])
                    classification_evaluation = classification_model_evaluation(data_splitted[3], y_pred, y_pred_prob)
                    print(classification_evaluation)
                    show_vizualization_classification(data_splitted[3], y_pred)
                else:
                    regression_evaluation = regresion_model_evaluation(data_splitted[3], y_pred)
                    print(regression_evaluation)
                    show_regresion_real_predicted(data_splitted[3], y_pred)
            else:
                print("You must select model before")
        elif main_choice == "6":
            pass
        elif main_choice == "0":
            continue
        else:
            print("Sorry but there is no such option")


if __name__ == '__main__':
    main()
