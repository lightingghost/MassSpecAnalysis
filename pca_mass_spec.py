from collections import defaultdict
import numpy as np

def get_peak_dict(img_data, resolution=0.1):
    '''img_data: n * m
    '''
    n, m = img_data.shape
    multiplier = 1 / resolution
    peaks_dict = defaultdict(int)
    for i in range(n):
        for j in range(m):
            mass_data = img_data[i, j]
            n_peaks = mass_data.shape[1]
            for k in range(n_peaks):
                peaks_dict[int(mass_data[0, k] * multiplier)] += 1

    peaks_vec = np.array(list(peaks_dict.keys()))
    return peaks_vec

def make_sample_vec(img_data, resolution=0.1):
    peaks_vec = get_peak_dict(img_data, resolution)
    n, m = img_data.shape
    multiplier = 1 / resolution
    nsamples = img_data.size
    nfeatures = peaks_vec.size
    peaks2idx = dict()
    for i in range(nfeatures):
        peaks2idx[peaks_vec[i]] = i
    sample_vec = np.zeros((nsamples, nfeatures))
    idx = 0
    for i in range(n):
        for j in range(m):
            mass_data = img_data[i, j]
            for k in range(mass_data.shape[1]):
                pk = int(mass_data[0, k] * multiplier)
                if pk in peaks2idx:
                    peak_idx = peaks2idx[pk]
                    sample_vec[idx, peak_idx] = mass_data[1, k]
            idx += 1
    return sample_vec

def get_component_composition(component, features, resolution, firstN=5):
    result = np.ndarray((2,firstN))
    index = component.argsort()[-firstN:][::-1]
    for i in range(firstN):
        result[0, i] = features[index[i]] * resolution
        result[1, i] = component[index[i]]
    return result

def pca_anal(img_data, resolution=0.1):
    sample_vec = make_sample_vec(img_data, resolution)
    peaks_vec = get_peak_dict(img_data, resolution)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    model = PCA()
    model.fit(sample_vec)

    variance_ratio = model.explained_variance_ratio_
    prin_comp = model.components_
    #output result
    output_file = open('result.txt', 'w')
    for num in range(3):
        output_file.write('#' + str(num + 1) + ' principle component, variance ratio =' + str(variance_ratio[num]) + ':\n')
        composition = get_component_composition(prin_comp[num,:], peaks_vec, resolution)

        output_file.write('peak:\t')
        for i in range(int(composition.size/2)):
            output_file.write(str(composition[0, i]) + '\t')
        output_file.write('\n')

        output_file.write('comp:\t')
        for i in range(int(composition.size/2)):
            output_file.write(str(composition[1, i]) + '\t')
        output_file.write('\n')

def main():
    path = 'D:\\Documents\\MyDocuments\\Zare\\FingerPrint\\03082016\\image_data.npy'
    img_data = np.load(path)
    pca_anal(img_data)

if __name__ == '__main__':
    main()
