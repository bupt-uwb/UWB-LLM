import numpy as np
from scipy.linalg import svd
from scipy.signal import firwin, convolve
from scipy.interpolate import interp1d

def signal_preprocess(signal, fs, fc, bw):
    M, N = signal.shape
    signal_mean = signal - np.mean(signal, axis=1, keepdims=True)
    signal_mean = signal_mean - np.mean(signal_mean, axis=0, keepdims=True)
    # signal_pca = principal_component_analysis(signal_mean)
    signal_dc = downconversion(signal_mean, fs, fc)
    signal_lp = lowpass_filter(signal_dc, fs, bw)
    signal_processed = np.abs(signal_lp)
    signal_processed[:,:50] = 0
    signal_processed[:,-10:] = 0
    return signal_processed


def lowpass_filter(signal,fs,bw):
    cutoff = 0.8 * bw / fs * 2  # 归一化截止频率（以 Nyquist 为1）

    # 设计FIR滤波器（窗函数法）
    numtaps = 201  # 滤波器阶数（可以调大改善频率分辨率）
    fir_coeff = firwin(numtaps, cutoff, window='hamming')

    def apply_filter(x):
        return np.convolve(x, fir_coeff, mode='same')
    if signal.ndim == 1:
        filtered_signal = apply_filter(signal)
    else:
        filtered_signal = np.apply_along_axis(apply_filter, axis=1, arr=signal)
    return filtered_signal

def principal_component_analysis(signal):
    U, S, Vt = svd(signal, full_matrices=False)
    sigma = S[0]
    u1 = U[:, 0].reshape(-1, 1)
    v1 = Vt[0, :].reshape(1, -1)
    noise = sigma * u1 @ v1
    signal_svd = signal - noise
    return signal_svd

def downconversion(signal, fs, fc):
    return signal * np.exp(-1j * 2 * np.pi * fc / fs * np.arange(signal.shape[1]))

def signal_normalize(signal):
    signal_norm =(signal - signal.min())/(signal.max() - signal.min())
    
    return signal_norm    
def signal_resize(signal_matrix, measured_length, step):
    x_old = np.linspace(measured_length[0], measured_length[1], signal_matrix.shape[1])
    x_new = np.arange(measured_length[0], measured_length[1], step=step)
    signal_resized = [interp1d(x_old, signal, kind='cubic')(x_new) for signal in signal_matrix]
    return np.array(signal_resized)


def signal_slice(signal, window_size, stride, start = 0):
    segments = []
    for i in range(start, len(signal) - window_size + 1, stride):
        segments.append(signal[i:i + window_size,:])
    return np.array(segments)


if __name__ == '__main__':
    pass