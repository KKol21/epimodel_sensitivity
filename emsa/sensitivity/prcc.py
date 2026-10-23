import warnings

import numpy as np
from scipy.stats import rankdata


def get_prcc_values(lhs_output_table):
    """
    Calculates the Partial Rank Correlation Coefficient (PRCC) values
    for the last column of an ndarray based on the preceding columns.

    Args:
        lhs_output_table (ndarray): Input ndarray containing the LHS samples and simulation results.

    Returns:
        ndarray: PRCC values for the last column. A coefficient is NaN if the
        output has no residual variation after conditioning on the other inputs.
    """
    table = np.asarray(lhs_output_table, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2 or table.shape[0] < 3:
        raise ValueError(
            "PRCC requires at least three samples, input columns, and an output column"
        )
    if not np.isfinite(table).all():
        raise ValueError("PRCC inputs and outputs must be finite")
    if np.any(np.ptp(table, axis=0) == 0):
        raise ValueError("PRCC is undefined for constant input or output columns")

    ranked = rankdata(table, axis=0)
    corr_mtx = np.corrcoef(ranked.T)
    if table.shape[1] == 2:
        return np.array([corr_mtx[0, 1]])
    if np.linalg.matrix_rank(corr_mtx) < len(corr_mtx):
        centered = ranked - ranked.mean(axis=0)
        inputs, output = centered[:, :-1], centered[:, -1]
        if np.linalg.matrix_rank(inputs) < inputs.shape[1]:
            raise ValueError("PRCC is undefined for rank-deficient input columns")
        # A perfectly ranked output can make the full correlation matrix singular
        # even with independent inputs. Correlate residuals for this case.
        result = np.full(inputs.shape[1], np.nan)
        for index in range(inputs.shape[1]):
            controls = np.delete(inputs, index, axis=1)
            x = inputs[:, index]
            x_residual = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
            y_residual = output - controls @ np.linalg.lstsq(controls, output, rcond=None)[0]
            denominator = np.linalg.norm(x_residual) * np.linalg.norm(y_residual)
            if np.linalg.norm(y_residual) > 1e-12 * np.linalg.norm(output):
                result[index] = np.dot(x_residual, y_residual) / denominator
        if np.isnan(result).any():
            warnings.warn(
                "Some PRCCs are undefined after conditioning; reported as NaN", RuntimeWarning
            )
        return np.clip(result, -1, 1)
    corr_mtx_inverse = np.linalg.inv(corr_mtx)

    parameter_count = table.shape[1] - 1

    prcc_vector = np.zeros(parameter_count)
    for w in range(parameter_count):  # compute PRCC btwn each param & sim result
        prcc_vector[w] = -corr_mtx_inverse[w, parameter_count] / np.sqrt(
            corr_mtx_inverse[w, w] * corr_mtx_inverse[parameter_count, parameter_count]
        )
    return np.clip(prcc_vector, -1, 1)
