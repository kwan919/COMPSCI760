from scipy import signal
import numpy as np
from sklearn import preprocessing


def normalize_data(v):
    """normalize the data
    PARAMATER:
        v: one data list
    RETURN:
        normalized data, in the shape of (len(v),1)
    """
    data = np.array(v).reshape(-1, 1)
    data = preprocessing.StandardScaler().fit_transform(data)
    return data


def add_WGN(v, snr, norm=True):
    """add Gaussian White Noise into data after nomalization
    PARAMATER:
        v: one data list
        snr: SNR
        norm: True, has been normalized
    RETURN:
        polluted data
    """
    data = np.array(v).reshape(-1, 1)
    if snr is None:
        return data
    if not norm:
        data = normalize_data(data)
    pwr_m = 1/(10**(snr/10))  # snr: 10*log(Ps/Pn) and Ps is 1(nomalized)
    n = np.random.normal(0, pwr_m, size=len(data)).reshape(-1, 1)
    data = np.add(n, data)
    return data


def add_rayleigh(v, snr, norm=True):
    """add Rayleigh Noise into data after nomalization
    PARAMATER:
        v: one data list
        snr: SNR
        norm: True, has been normalized
    RETURN:
        polluted data
    """
    data = np.array(v).reshape(-1, 1)
    if snr is None:
        return data
    if not norm:
        data = normalize_data(data)
    m = 10**(snr/20)  # snr: 20*log(As/An) and Ps is 1(nomalized)
    n = np.random.rayleigh(size=len(data)).reshape(-1, 1)
    n = normalize_data(n)*m
    data = np.add(n, data)
    return data
