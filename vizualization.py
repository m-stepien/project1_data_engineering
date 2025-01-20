import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np


def show_histogram(data, column_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=100, density=True, alpha=0.5, histtype='stepfilled')
    ax.set_title(f"Histogram dla {column_name}")
    plt.show()
    return fig


def show_all_histograms(data):
    num_columns = len(data.columns)
    cols = min(5, num_columns)
    rows = (num_columns // cols) + (num_columns % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)

    if num_columns == 1:
        axes = [axes]

    axes = axes.flatten() if num_columns > 1 else axes

    for i, column in enumerate(data.columns):
        axes[i].hist(data[column].dropna(), bins=100, density=True, alpha=0.5, histtype='stepfilled')
        axes[i].set_title(f"Histogram dla {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Density")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()
    return fig


def show_scatter_plot(x, y, x_name, y_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.7)
    ax.set_title(f"{x_name} {y_name}", fontsize=14)
    ax.set_xlabel(x_name, fontsize=12)
    ax.set_ylabel(y_name, fontsize=12)
    ax.ticklabel_format(style='plain', axis='both')
    plt.show()
    return fig

def show_heatmap(corr_matrix):
    fig, ax = plt.subplots(constrained_layout=True)  # Tworzymy figurę i przypisujemy oś
    annot = True
    shape = corr_matrix.shape[0]

    if shape <= 5:
        fmt, font_size = ".3f", 12
    elif shape <= 12:
        fmt, font_size = ".2f", 10
    elif shape <= 20:
        fmt, font_size = ".1f", 8
    else:
        fmt, font_size = ".1f", 8
        annot = False

    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', fmt=fmt, annot_kws={"size": font_size}, ax=ax)

    ax.set_title('Macierz korelacji', fontsize=14)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    plt.show()
    return fig


def show_missing_data(data):
    fig, ax = plt.subplots(constrained_layout=True)
    missing_percent = (data.isnull().sum() / len(data)) * 100
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_title('Procent brakujących wartości dla każdej kolumny', fontsize=14)
    ax.set_xlabel('Kolumny', fontsize=12)
    ax.set_ylabel('Procent braków danych', fontsize=12)
    missing_percent.plot(kind='bar', color='red', ax=ax)
    plt.show()
    return fig


def data_distribution(data, name=None):
    if data.dtypes.kind in ['i', 'u', 'f', 'c']:
        return data_distribution_numerical_data(data, name)
    else:
        return data_distribution_categorical_data(data, name)


def data_distribution_numerical_data(data, name):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.boxplot(x=data)
    ax.set_title('Rozkład zmiennych', fontsize=14)
    ax.set_ylabel('Wartości', fontsize=12)
    ax.set_xlabel(name, fontsize=12)
    plt.show()
    return fig


def data_distribution_categorical_data(data, name):
    category_counts = data.value_counts()
    fig, ax = plt.subplots(constrained_layout=True)
    category_counts.plot(kind='bar', color='gray', alpha=0.8, ax=ax)
    ax.set_title(f'Liczba wystąpień kategorii w {name}', fontsize=14)
    ax.set_xlabel('Kategorie', fontsize=12)
    ax.set_ylabel('Liczba wystąpień', fontsize=12)
    plt.show()
    return fig


def show_regresion_real_predicted(y, y_pred):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(y, y_pred, alpha=0.7, label='Rzeczywiste wartości vs przewidywane')
    ax.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Idealne dopasowanie')
    ax.set_xlabel('Wartości rzeczywiste')
    ax.set_ylabel('Wartości przewidywane')
    ax.set_title('Rzeczywiste vs Przewidywane')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
    return fig


def show_vizualization_classification(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    labels = np.unique(np.concatenate((y, y_pred)))
    fig, ax = plt.subplots(constrained_layout=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    ax.set_title('Macierz Pomyłek')
    plt.show()
    return fig


