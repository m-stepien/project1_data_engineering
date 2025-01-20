import pandas as pd
import numpy as np

EXCEL_EXTENSIONS = ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]
CSV_EXTENSIONS = ["csv", "tsv"]
JSON_EXTENSIONS = ["json"]
DEFAULT_ATRIBUTES = {"csv": {"sep": ",", "decimal": ".", "index_col": None, "skiprows": None,
                             "header": "infer", "usecols": None},
                     "json": {"orient": None, "typ": None},
                     "excel": {"sheet_name": 0, "header": 0}}


def load_from_file(filename: str, atributes: dict):
    extension = filename.split(".")[-1]
    dataframe = None
    filled_atributes = add_default_param_for_file_extension(extension, atributes)
    print(filled_atributes)
    if extension in JSON_EXTENSIONS:
        dataframe = pd.read_json(filename)
    elif extension in CSV_EXTENSIONS:
        dataframe = pd.read_csv(filename, sep=filled_atributes.get("sep"), decimal=filled_atributes.get("decimal"),
                                index_col=filled_atributes.get("index_col"), skiprows=filled_atributes.get("skiprows"),
                                header=filled_atributes.get("header"), usecols=filled_atributes.get("usecols"))
    elif extension in EXCEL_EXTENSIONS:
        dataframe = pd.read_excel(filename)
    return dataframe


def add_default_param_for_file_extension(file_extension: str, atributes: dict):
    result = None
    if file_extension in CSV_EXTENSIONS:
        result = add_default_param("csv", atributes)
    elif file_extension in JSON_EXTENSIONS:
        result = add_default_param("json", atributes)
    elif file_extension in EXCEL_EXTENSIONS:
        result = add_default_param("xls", atributes)
    return result


def add_default_param(extension_category, atributes):
    set_of_filled_params_names = set(atributes.keys())
    set_of_required_params_names = set(DEFAULT_ATRIBUTES.get(extension_category).keys())
    missing_params_names = set_of_required_params_names.difference(set_of_filled_params_names)
    for param in missing_params_names:
        atributes[param] = DEFAULT_ATRIBUTES.get("csv").get(param)
    return atributes


def get_atributes_list_for_file_extension(extension):
    return DEFAULT_ATRIBUTES.get(extension).keys()
