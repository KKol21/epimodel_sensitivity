import numpy as np


def get_prcc_values(lhs_output_table):
    """
    Calculates the Partial Rank Correlation Coefficient (PRCC) values
    for the last column of an ndarray based on the preceding columns.

    Args:
        lhs_output_table (ndarray): Input ndarray containing the LHS samples and simulation results.

    Returns:
        ndarray: PRCC values for the last column.
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
