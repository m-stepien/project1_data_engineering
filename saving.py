import pandas as pd
import pickle
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def save_process_data_to_csv(df, filename="data"):
    df.to_csv(filename + ".csv", index=False)


def save_result_to_json(to_save, filename="data"):
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)


def save_model(model, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(model, f)


def save_to_pdf(figures, filename="raport"):
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)