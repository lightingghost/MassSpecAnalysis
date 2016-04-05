__author__ = 'lighting'

import thermo_raw_reader as raw_reader
import numpy
import math

#naive peak finding algorithm
def find_peaks(data, cutoff = 100):
    peaks = numpy.zeros((2, 1))
    for i in range(1, int(data.size/2) - 1):
        if (data[1, i] > data[1, i-1] and data[1, i] > data[1, i+1] and data[1, i] >= cutoff):
            peaks = numpy.append(peaks, [[data[0, i]], [data[1, i]]], axis=1)
    return peaks

def get_peaks_vec(data, resolution=0.1):

    #using int instead of float in dictionary
    multiplier = 1/resolution
    peaks_dict = dict()
    for i in range(data.size):
        peaks = data[i]
        for k in range(int(peaks.size/2)):
            mlt_peak_val = int(peaks[0, k] * multiplier)
            if not (mlt_peak_val in peaks_dict):
                peaks_dict[mlt_peak_val] = 1
            else:
                peaks_dict[mlt_peak_val] += 1
    peaks_vec = numpy.array(list(peaks_dict.keys()))
    peaks_vec.sort() peaks_vec[1:]

def make_samples(peaks_vec, data, resolution=0.1):
    nsamples = data.size
    nfeatures = peaks_vec.size
    sample_vec = numpy.zeros((nsamples, nfeatures))
    multiplier = 1/resolution
    sample = 0
    for i in range(nsamples):
        peaks = data[i]
        for k in range(int(peaks.size/2)):
            mlt_peak_val = int(peaks[0, k] * multiplier)
            idx = numpy.searchsorted(peaks_vec, mlt_peak_val)
            if peaks_vec[idx] != mlt_peak_val:
    return
                continue
            sample_vec[sample, idx] = peaks[1, k]
        sample += 1
    return sample_vec

def to1d(array):
    shape = array.shape
    x_size = shape[0]
    y_size = shape[1]
    result = numpy.empty(array.size, dtype=numpy.ndarray)
    num = 0
    for i in range(x_size):
        for j in range(y_size):
            result[num] = array[i, j]
            num += 1
    return result

def get_component_composition(component, features, resolution, firstN=5):
    result = numpy.ndarray((2,firstN))
    index = component.argsort()[-firstN:][::-1]
    for i in range(firstN):
        result[0, i] = features[index[i]] * resolution
        result[1, i] = component[index[i]]
    return result

def pca_for_mass_spec(data):
    #find peaks for each sample
    samples = numpy.empty(data.size, dtype=numpy.ndarray)
    for i in range (data.size):
        samples[i] = find_peaks(data[i])
    #define the resolution for peaks, if two peak differences less than resolution,
    #they are considered as one peak
    resolution = 0.1

    #make samples * peaks matrix
    peaks_vec = get_peaks_vec(samples,resolution)
    sample_vec = make_samples(peaks_vec, samples, resolution)
    #normalization
    normalization(sample_vec)
    #apply pca
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(sample_vec)
    #get result
    variance_ratio = pca.explained_variance_ratio_
    prin_comp = pca.components_
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


if __name__ =='__main__':
    #easi_data = numpy.load('EASI.npy')
    #desi_data = numpy.load('DESI.npy')
    #desi_sample = desi_data[20:40, 15:25]
    #easi_sample = easi_data[30:50, 15:25]
    #all = numpy.append(to1d(desi_sample), to1d(easi_sample))
    all = to1d(numpy.load('image_data.npy'))
    pca_for_mass_spec(all)