import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
cimport cython

from stcal.ramp_fitting.ols_cas22._fit_one_ramp import fit_one_ramp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float correction_factor(int i, int j, float [:] t_bar):
    """Compute the correction factor

    Parameters
    ----------
    i : int
        The index of the first read in the segment
    j : int
        The index of the last read in the segment
    t_bar : float 
    """
    cdef float denom = t_bar[-1] - t_bar[0]

    if j - i == 1:
        return (1 - (t_bar[i + 1] - t_bar[i]) / denom) ** 2
    else:
        return (1 - 0.75 * (t_bar[i + 2] - t_bar[i]) / denom) ** 2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float delta_var(
    int i, int j, float [:] t_bar, float [:] tau, float [:] n_reads, float read_noise, float slope):

    return (
            (
                read_noise * (1 / n_reads[i] + 1 / n_reads[j]) + 
                slope * (tau[i] + tau[j] - np.min(t_bar[i], t_bar[j])) * correction_factor(i, j, t_bar)
            ) / ((t_bar[j] - t_bar[i]) ** 2)
        )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float stat(
    int i, int j, float[:] resultants, float [:] t_bar, float [:] tau,
    float [:] n_reads, float read_noise, float slope):
    cdef float delta = ((resultants[j] - resultants[i]) / (t_bar[j] - t_bar[i])) - slope

    return delta / sqrt(delta_var(i, j, t_bar, tau, n_reads, read_noise, slope))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float *statistics(
        float [:] resultants, float [:] t_bar, float [:] tau, float [:] n_reads, float read_noise, float slope
    ):
    cdef int n_stats = len(n_reads), i

    cdef float *stats = <float *> malloc(n_stats * sizeof(float))
    cdef float stat_1, stat_2

    for i in range(n_stats):
        stat_1 = stat(i, i + 1, resultants, t_bar, tau, n_reads, read_noise, slope)
        stat_2 = stat(i, i + 2, resultants, t_bar, tau, n_reads, read_noise, slope)

        stats[i] = np.max(stat_1, stat_2)

    return stats


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef inline (float, float, float) jump_detection(
#         float [:] resultants, int start, int end, float read_noise,
#         float [:] t_bar, float [:] tau, int [:] n_reads):

