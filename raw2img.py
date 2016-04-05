# improvment:
# use peak value instead of all values to save space

import numpy
import os


def find_peak_value(mass_data, peak, tolerance = 0.1):
    for i in range(int(mass_data.size/2)):
        if mass_data[0, i] < peak + tolerance and mass_data[0, i] > peak - tolerance:
            return mass_data[1, i]
    return 0

class Image:
    def __init__(self, length=1, width=1):
        self.length = length #num of scans per file
        self.width = width #num of files
        self.image_data = numpy.empty((self.length, self.width), dtype=numpy.ndarray)


    def load_raw(self, path):
        import thermo_raw_reader as reader

        files = os.listdir(path)
        raw_files = [name for name in files if (name.find('.raw') != -1 and name.find('.cfg') == -1)]

        self.width = len(raw_files)
        rawfile = reader.RawFile(path + raw_files[0])
        self.length = rawfile.get_num_spectra() - 1
        self.image_data = numpy.empty((self.length, self.width), dtype=numpy.ndarray)

        y = 0
        for filename in raw_files:
            filepath = path + filename
            rawfile = reader.RawFile(filepath)
            scan_num = rawfile.get_num_spectra()
            if scan_num < self.length:
                print('Not Enough Data Points' + filename)
            for x in range(self.length):
                mass_data = rawfile.get_mass_list(x + 1)
                self.image_data[x, y] = self.find_peaks(mass_data)
            y = y + 1
            rawfile.close()
        return self.image_data

    def save_image_data(self, path):
        numpy.save(path, self.image_data)

    def load_image_data(self, filepath):
        self.image_data = numpy.load(filepath)
        self.length = self.image_data.shape[0]
        self.width = self.image_data.shape[1]
        return self.image_data

    def get_image(self, peak):
        self.ms_image = numpy.zeros((self.length, self.width))
        for x in range(self.length):
            for y in range(self.width):
                self.ms_image[x, y] = find_peak_value(self.image_data[x, y],peak)
        return self.ms_image

    def plot_image(self, method='contour'):
        import matplotlib.pyplot as plt
        x = range(self.length)
        y = range(self.width)
        max = self.ms_image.max()
        min = self.ms_image.min()
        plt.figure()
        cmap = plt.cm.get_cmap('rainbow')
        plt.contourf(y, x, self.ms_image,numpy.arange(min,max/3,(max/3-min)/10), cmap=cmap ,extend='both')
        plt.show()

    def find_peaks(self, data, cutoff_percent = 0.05):
        peaks = numpy.zeros((2, 1))
        cutoff = cutoff_percent * (data[1, :].max())
        for i in range(1, int(data.size/2) - 1):
            if (data[1, i] > data[1, i-1] and data[1, i] > data[1, i+1] and data[1, i] >= cutoff):
                peaks = numpy.append(peaks, [[data[0, i]], [data[1, i]]], axis=1)
        return peaks

def validation():
    r = numpy.loadtxt('result.txt')
    path = 'D:\\Documents\\MyDocuments\\Zare\\FingerPrint\\03082016\\'
    image = Image()
    peaks = numpy.squeeze(r[:,0])
    image_data = image.load_image_data(path+'image_data.npy')
    for i in range(peaks.size):
        img = image.get_image(peaks[i])
        max = img.max()
        min = img.min()
        import matplotlib.pyplot as plt
        plt.figure(i)
        cmap = plt.cm.get_cmap('rainbow')
        plt.contourf(img, numpy.arange(min,max/3,(max/3-min)/10), cmap=cmap ,extend='both')
        plt.title(str(peaks[i]))
        plt.show()

def get_img():
    length = 3
    width = 4
    path = 'D:\\Documents\\MyDocuments\\Zare\\FingerPrint\\03082016\\'

    image = Image()
    #image.load_raw(path)
    #image.save_image_data(path+'image_data.npy')
    image_data = image.load_image_data(path+'image_data.npy')
    #image_112 = image.get_image(111.9)
    #numpy.savetxt('image_112.txt',image_112)
    #image_126 = image.get_image(125.9)
    #numpy.savetxt('image_128.txt',image_126)
    img = image.get_image(66.16)
    #image.get_image(311.1)
    #image.get_image(808.5)
    #image.get_image(303.2)
    #image.get_image(327.2)
    #image.get_image(279.2)
    #image_195 = image.get_image(195.09)
    #numpy.savetxt('image_195.txt', image_195)
    image.plot_image()

if __name__ == '__main__':
    #validation()
    get_img()