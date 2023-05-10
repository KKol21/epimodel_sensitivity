from matplotlib import pyplot as plt
import numpy as np


def get_prcc_values(lhs_output_table):
    """
    Creates the PRCC values of last column of an ndarray depending on the columns before.
    :param lhs_output_table: ndarray
    :return: ndarray
    """
    ranked = (lhs_output_table.argsort(0)).argsort(0)
    corr_mtx = np.corrcoef(ranked.T)
    if np.linalg.det(corr_mtx) < 1e-50:  # determine if singular
        corr_mtx_inverse = np.linalg.pinv(corr_mtx)  # may need to use pseudo inverse
    else:
        corr_mtx_inverse = np.linalg.inv(corr_mtx)

    parameter_count = lhs_output_table.shape[1] - 1

    prcc_vector = np.zeros(parameter_count)
    for w in range(parameter_count):  # compute PRCC btwn each param & sim result
        prcc_vector[w] = -corr_mtx_inverse[w, parameter_count] / \
                         np.sqrt(corr_mtx_inverse[w, w] *
                                 corr_mtx_inverse[parameter_count, parameter_count])
    return prcc_vector


def generate_prcc_plot(params, target_var, prcc: np.ndarray, filename: str):
    prcc = np.round(prcc, 3)
    sorted_idx = np.abs(prcc).argsort()[::-1]
    prcc = prcc[sorted_idx]
    if target_var == "i_max":
        target_var = "fertőzöttek száma a csúcson"
    elif target_var == "ic_max":
        target_var = "intenzív osztályra kerültek száma a csúcson"
    elif target_var == "d_max":
        target_var = "halottak száma"
    plt.title(f"PRCC értékei a vakcinák elosztásának, célváltozó: {target_var}", fontsize=15, wrap=True)

    ys = range(len(params))[::-1]
    # Plot the bars one by one
    for y, value in zip(ys, prcc):
        plt.broken_barh(
            [(value if value < 0 else 0, abs(value))],
            (y - 0.4, 0.8),
            facecolors=['white', 'white'],
            edgecolors=['black', 'black'],
            linewidth=1,
        )

        if value != 0:
            x = (value / 2) if np.abs(value) >= 0.2 else (- np.sign(value) * 0.1)
        else:
            x = -0.1
        plt.text(x, y - 0.01, str(value), va='center', ha='center')

    plt.axvline(0, color='black')

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    axes = plt.gca()  # (gca = get current axes)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.xaxis.set_ticks_position('top')

    # Make the y-axis display the params
    labels = list(map(get_age_group, params[sorted_idx]))
    plt.yticks(ys, labels)

    # Set the portion of the x- and y-axes to show
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1, len(params))
    # plt.text()
    plt.savefig(f'../sens_data/plots/prcc_tornado_plot_{filename}.pdf',
                format="pdf", bbox_inches='tight')
    plt.show()


def get_age_group(param):
    age_start = int(param.split('_')[2]) * 5
    return f'{age_start}-{age_start + 5} ' if age_start != 75 else '75+ '
