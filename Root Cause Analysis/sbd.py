import math
import numpy as np
import multiprocessing
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
from sklearn.base import ClusterMixin, BaseEstimator

def _ncc_c_3dim(data):
    x, y = data[0], data[1]
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    fft_size = 1 << (2 * x_len - 1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den

def sbd_distance(x, y):
    ncc = _ncc_c_3dim([x, y])
    # for i in range(len(ncc)):
    #     if i < 7 or i > 11:
    #         ncc[i] = -2
    idx = np.argmax(ncc)
    # yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))
    ncc_max = np.max(ncc)
    return 1 - ncc_max, idx