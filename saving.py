import pandas as pd
import pickle


def save_process_data_to_csv(df, filename="data"):
    df.to_csv(filename + ".csv", index=False)


def save_process_data_to_json(df, filename="data"):
    df.to_json(filename + ".json")


def save_model(model, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(model, f)
