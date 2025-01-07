import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def show_histogram(data):
    plt.hist(data, bins=100, density=True, alpha=0.5, histtype='stepfilled')
    plt.show()


def show_scatter_plot(x, y, x_name, y_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(name, fontsize=14)
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel(y_name, fontsize=12)
    plt.ticklabel_format(style='plain', axis='both')
    plt.show()


def show_heatmap(corr_matrix):
    plt.figure(constrained_layout=True)
    annot = True
    shape = corr_matrix.shape[0]
    if shape <= 5:
        fmt = ".3f"
        font_size = 12
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    elif shape <= 12:
        fmt = ".2f"
        font_size = 10
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    elif shape <= 20:
        fmt = ".1f"
        font_size = 8
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    else:
        fmt = ".1f"
        font_size = 8
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        annot = False
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', fmt=fmt, annot_kws={"size": font_size})
    plt.title('Macierz korelacji', fontsize=14)
    plt.show()
