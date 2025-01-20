import pandas as pd
import pickle
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfMerger


def save_process_data_to_csv(df, filename="data"):
    df.to_csv(filename + ".csv", index=False)


def save_result_to_json(to_save, filename="data"):
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)


def save_model(model, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(model, f)


def save_to_pdf(figures, stats_dict, filename="raport"):
    save_dict_to_pdf(stats_dict, "temp2.pdf")
    with PdfPages("temp.pdf") as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    merge_pdfs(filename+".pdf", ["temp2.pdf", "temp.pdf"])


def save_dict_to_pdf(stats_dict, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    for main_category, sub_dict in stats_dict.items():
        elements.append(Paragraph(f"<b>{main_category}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        table_data = [["Nazwa", "Wartosc"]]
        for sub_category, values in sub_dict.items():
            if isinstance(values, dict):
                table_data.append([f"{sub_category}", ""])
                for key, value in values.items():
                    table_data.append([f"  {key}", f"{value:.2f}" if isinstance(value, (int, float)) else str(value)])
            else:
                table_data.append([sub_category, f"{values:.2f}" if isinstance(values, (int, float)) else str(values)])

        table = Table(table_data, colWidths=[250, 150])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
    doc.build(elements)


def merge_pdfs(output_filename, pdf_filenames):
    merger = PdfMerger()
    for pdf in pdf_filenames:
        merger.append(pdf)
    merger.write(output_filename)
    merger.close()
