from keras.models import load_model
import keras.backend as K
import numpy as np
from skimage import io
from scipy.signal import convolve2d
from scipy import ndimage
import tifffile
import copy

def norm(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img) + 1.0e-10)

def filter_singularity( image, denoised_image, radius=3, threshold=60000 ):
    denoised_image = np.squeeze( denoised_image )
    image = np.squeeze( image )
    mask1 = denoised_image > threshold
    mask = np.asarray( mask1, dtype='float32' )
    denoised_image = norm( denoised_image )
    tmp = copy.deepcopy( denoised_image )
    if np.sum( mask ) > 0:
        new_mask = convolve2d(mask, np.ones((radius,radius)), mode='same', boundary='symm')
        denoised_image[new_mask>0.0] = 0.0

        smooth_denoised = ndimage.gaussian_filter( image, 4 )
        smooth_denoised = norm( smooth_denoised )

        mask_intensity = smooth_denoised > 0.5
        denoised_image[mask_intensity] = tmp[mask_intensity]

    return np.asarray( denoised_image*65535, dtype='uint16' )

# adjust batch size according to the image size
def predict(image_path, prediction_path=None, enable_padding=True):
    prediction_path = prediction_path or (image_path + '_denoised.tiff')

    # load Low Pass Filter Model
    lpf = load_model('../../data/lpf.model')
    lpf._make_predict_function()

    # load Denoiser Model
    generator = load_model('../../data/denoising.model')
    generator._make_predict_function()

    # load experimental data
    image = np.squeeze( np.asarray(io.imread(image_path), dtype='float32') )
    if len(image.shape) == 2:
        image = image.reshape((1, ) + image.shape)
    number, row, col = image.shape
    image = norm( image )

    # 512 - 2
    lpf_batch_size, generator_batch_size = 8, 2
    if max( row, col) > 512:
        lpf_batch_size, generator_batch_size = 1, 1

    # symmetric padding of experimental data
    if enable_padding:
        image = np.pad(image, ((0, 0), (128, 128), (128, 128)), 'symmetric')
    image = image.reshape(image.shape + (1, ))

    # low-pass filter process
    proceed_image = lpf.predict(image, batch_size=lpf_batch_size, verbose=1)

    # correct contrast
    for idx in range(number):
        for jdx in range(4):
            proceed_image[idx, :, :, jdx] = norm(proceed_image[idx, :, :, jdx])

    # denoising process
    prediction = generator.predict(proceed_image, batch_size=generator_batch_size, verbose=1)

    # unpadding
    if enable_padding:
        image = image[:, 128:128 + row, 128:128 + col, :]
        result = prediction[:, 128:128 + row, 128:128 + col, :]

    prediction_result = norm(result)
    prediction_result = np.squeeze( np.asarray( prediction_result * (256 * 256 - 1), dtype='uint16') )
    image = norm(image)
    image = np.squeeze( np.asarray( image * (256 * 256 - 1), dtype='uint16') )

    for idx in range( number ):
        prediction_result[idx] = filter_singularity( image[idx], prediction_result[idx], radius=5 );

    tifffile.imsave(prediction_path, prediction_result)

    K.clear_session()

if __name__ == '__main__':
    predict( '../../data/s21.tif' )
    predict( '../../data/s3.tif' )
