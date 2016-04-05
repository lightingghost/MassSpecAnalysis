from collections import defaultdict
import numpy as np
import sklearn as sl

def get_peak_dict(img_data, resolution=0.1, freq_thresh=0.02):
    '''img_data: n * m
    '''
    n, m = img_data.shape
    multiplier = 1 / resolution
    num_thresh = n * m * freq_thresh
    peaks_dict = defaultdict(int)
    for i in range(n):
        for j in range(m):
            mass_data = img_data[i, j]
            n_peaks = mass_data.shape[1]
            for k in range(n_peaks):
                peaks_dict[int(mass_data[0, k] * multiplier)] += 1

    peaks_vec = np.array([key for key in peaks_dict.keys() if peaks_dict[key] > num_thresh])[1:]
    peak2idx = dict()
    idx2peak = dict()
    for i in range(peaks_vec.size):
        peak2idx[peaks_vec[i]] = i
    for key in peak2idx.keys():
        idx2peak[peak2idx[key]] = key
    return peaks_vec, peak2idx, idx2peak

def make_sample_vec(img_data, peak2idx, resolution=0.1):
    n, m = img_data.shape
    multiplier = 1 / resolution
    nsamples = img_data.size
    nfeatures = len(peak2idx)

    sample_vec = np.zeros((nsamples, nfeatures))
    idx = 0
    for i in range(n):
        for j in range(m):
            mass_data = img_data[i, j]
            for k in range(mass_data.shape[1]):
                pk = int(mass_data[0, k] * multiplier)
                if pk in peak2idx:
                    peak_idx = peak2idx[pk]
                    sample_vec[idx, peak_idx] = mass_data[1, k]
            idx += 1
    return sample_vec

def make_label(X, rel_peak, peak2idx, resolution=0.1):
    pk = int(rel_peak * 1 / resolution) + 1
    idx = peak2idx[pk]
    avg = np.mean(X[:, idx]) * 1.1
    Y = (X[:, idx] > avg).astype(int)
    return Y

def coef_to_peak(coef, idx2peak, resolution=0.1):
    idxs = np.argsort(coef)[::-1]
    peaks = np.zeros((2, coef.size))
    for i in range(coef.size):
        peaks[0, i] = idx2peak[idxs[i]] * resolution
        peaks[1, i] = coef[idxs[i]]
    return peaks

def ft_selec(img_data, rel_peak, resolution=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize
    peaks_vec, peak2idx, idx2peak = get_peak_dict(img_data)
    X = make_sample_vec(img_data, peak2idx, resolution)
    #X = normalize(X, norm='l1')
    Y = make_label(X, rel_peak, peak2idx, resolution)

    model = LogisticRegression(penalty='l1', n_jobs=4, C=0.00002)
    model.fit(X, Y)
    coef = np.squeeze(model.coef_)
    peaks = coef_to_peak(coef, idx2peak)
    output_file = open('result.txt', 'w')
    for i in range(10):
        output_file.write(str(peaks[0, i]) + '\t' + str(peaks[1, i]) + '\n')
    output_file.close()

def main():
    path = 'D:\\Documents\\MyDocuments\\Zare\\FingerPrint\\03082016\\image_data.npy'
    img_data = np.load(path)

    ft_selec(img_data, 255.2)

if __name__ == '__main__':
    main()

