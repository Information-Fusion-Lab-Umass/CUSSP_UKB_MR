import cv2
import skimage
from skimage.color import gray2rgb as sk_gray2rgb
from skimage.exposure import rescale_intensity as sk_rescale_intensity
from skimage.transform import resize as sk_resize
import numpy as np
import matplotlib.pyplot as plt

from cardiac_segmentation.dataset.processing import pad_data
from .data import load_niigz


def gray2rgb(image):
    if image.shape[-1] == 3:
        return image

    if image.ndim > 2:
        return np.array([gray2rgb(frame) for frame in image])

    return sk_gray2rgb(image)

def rescale_intensity(image, dtype):
    if image.shape[-1] == 3 or image.dtype == dtype:
        return image

    if image.ndim > 2:
        return  np.array([rescale_intensity(frame, dtype) for frame in image])

    return sk_rescale_intensity(image, out_range=dtype).astype(dtype)


def resize(image, ratio):
    if (image.shape[-1] == 3 and image.ndim > 3) or \
       (image.shape[-1] > 3 and image.ndim > 2):
        return np.array([resize(frame, ratio) for frame in image])

    resize_shape = (image.shape[0] * ratio,
                    image.shape[1] * ratio)
    return sk_resize(image, resize_shape)

def pack_sequences(sequences, labels, shape=(210,212), 
                   n_rows=4, n_cols=4, column_first=True, n_frames=50,
                   readjust=True):

    video = np.zeros((n_frames, shape[0]*n_rows, shape[1]*n_cols, 3)).astype(np.uint8)
    padding = pad_data(shape)
    
    for i, label in enumerate(labels):
        # loop over pids
        pid_sequences = sequences[i]
        for j, sequence in enumerate(pid_sequences):
            # loop over different sequences for a pid

            # readjust sequnce frame size that are too small
            reshape_ratio = min([shape[0] // sequence.shape[1] - 1, 
                                 shape[1] // sequence.shape[2] - 1])
            if reshape_ratio >= 2 and readjust:
                sequence = resize(sequence, reshape_ratio)

            # padding, rescale_intensity, and convert to RGB
            padded_sequence = padding(sequence)
            rescaled_sequence = rescale_intensity(padded_sequence, np.uint8)
            rgb_sequence = gray2rgb(rescaled_sequence)
            
            for frame in rgb_sequence:
                cv2.putText(frame, f"LABEL: {label}", 
                            (14,14), 1, 1.0,(0,255,0),1)
            
            if column_first:
                col_idx, row_idx = i, j
            else:
                row_idx, col_idx = i, j
                
            row_start = row_idx * shape[0]
            row_end = (row_idx+1)*shape[0]
            col_start = col_idx * shape[1]
            col_end = (col_idx+1)*shape[1]
            
            video[:, row_start:row_end, col_start:col_end] = rgb_sequence
            
    return video
        

def pack_4x4_images(pids, ext="la_4ch", shape=(210,212)):
    n_rows, n_cols = 4, 4
    frame = np.zeros((50, shape[0]*n_rows, shape[1]*n_cols, 3)).astype(np.uint8)
    padding = pad_data(shape)

    for i, pid in enumerate(pids):
        image = load_niigz(pid, ext)
        label = csv.query(f"ID=={pid}").LABEL.item()
        row_idx, col_idx = i % n_rows, i // n_cols
        padded_image = gray2rgb(rescale_intensity(padding(image), np.uint8))

        for padded_frame in padded_image:
            cv2.putText(padded_frame, f'LABEL: {label}',
                        (14,14), 1, 1.0,(0,255,0),1)

        row_start = row_idx * shape[0]
        row_end = (row_idx+1)*shape[0]
        col_start = col_idx * shape[1]
        col_end = (col_idx+1)*shape[1]

        frame[:, row_start:row_end, col_start:col_end] = padded_image

    return frame


def histogram_equalize(sequence, mask=None, n_bins=None, pixel_range=None):
    if mask is None:
        mask = np.ones_like(sequence).astype(bool)

    masked_sequence = np.ma.array(sequence, mask=~mask)
    
    if n_bins == None:
        #pixel_range = [int(sequence.min()), 
        #               int(sequence.max())]
        pixel_range = [int(masked_sequence.min()), 
                       int(masked_sequence.max())]
        n_bins = pixel_range[1] - pixel_range[0] + 1
    
    hist, bins = np.histogram(masked_sequence.compressed().flatten(),  n_bins, pixel_range)
    cdf = hist.cumsum()
    cdf_ma = np.ma.masked_equal(cdf, 0)
    cdf_ma = (cdf_ma - cdf_ma.min()) * (n_bins-1) / (cdf_ma.max() - cdf_ma.min())
    mapping = np.ma.filled(cdf_ma, 0).astype(int)
    
    #return mapping, pixel_range
    return hist_equal_map(mapping, pixel_range)

def histogram_equalize_test(sequence, mask=None, n_bins=None, pixel_range=None):
    if mask is None:
        mask = np.ones_like(sequence).astype(bool)

    masked_sequence = np.ma.array(sequence, mask=~mask)
    
    if n_bins == None:
        pixel_range = [int(sequence.min()), 
                       int(sequence.max())]
        n_bins = pixel_range[1] - pixel_range[0] + 1
    
    hist, bins = np.histogram(masked_sequence.compressed().flatten(),  n_bins, pixel_range)
    cdf = hist.cumsum()
    cdf_ma = np.ma.masked_equal(cdf, 0)
    cdf_ma = (cdf_ma - cdf_ma.min()) * (n_bins-1) / (cdf_ma.max() - cdf_ma.min())
    mapping = np.ma.filled(cdf_ma, 0).astype(int)
    
    #return mapping, pixel_range
    return hist_equal_map(mapping, pixel_range)


class hist_equal_map(object):
    def __init__(self, mapping, pixel_range):
        self.mapping = mapping
        self.pixel_range = pixel_range

    def __call__(self, images):
        images_ma = np.ma.masked_less(images, self.pixel_range[0])
        images_ma = np.ma.masked_greater(images_ma, self.pixel_range[1])
        images_ma -= self.pixel_range[0]

        images_ma = images_ma.filled(0)
        mapped_images = self.mapping[images_ma.astype(int)]

        return mapped_images
    


def plot_hist(image, max_value=1024, pixel_range=[0, 1024]):
    if max_value == None:
        max_value = int(image.max()) + 1
        pixel_range = [int(image.min()), 
                       int(image.max() + 1)]
    
    hist, bins = np.histogram(image.flatten(), max_value, pixel_range)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    plt.figure(figsize=(12,8))
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), max_value, pixel_range, color='r')
    plt.xlim(pixel_range)
    plt.legend(('cdf','histogram'), loc='upper left')
    plt.show()
