from keras.models import model_from_json
from keras import backend
import imageio
import numpy as np
import os

def dump_all_images( parent_path, arrays ):
    n, row, col, ch = arrays.shape
    arrays = np.squeeze( arrays )
    arrays = 255.0 * ( arrays - np.amin(arrays) ) / ( np.amax(arrays) - np.amin(arrays) + 1.0e-10 )
    for idx in range( n ):
        file_name = f'{parent_path}_{idx}.png'
        imageio.imsave( file_name, np.asarray(arrays[idx], dtype='uint8') )
        print( f'{file_name} dumped', end='\r' )
    print(' ')

def make_prediction( config ):
    model_json_path = config['model_json_path']
    model_weights_path = config['model_weights_path']
    input_path = config['input_path']
    output_path = config['output_path']
    learning_phase_fix = config['learning_phase_fix']
    batch_size = config['batch_size']
    dump_inputs = config['dump_inputs']

    if learning_phase_fix:
        backend.set_learning_phase( 1 )

    with open(model_json_path, 'r') as file:
        model = model_from_json( file.read() )

    model.load_weights(model_weights_path)
    model.save( '../../data/diffuse_reflection_single_gpu.model' )
    model.summary()

    input_data = np.load( input_path )
    output_data_high_frequency, *_ = model.predict( input_data, batch_size=batch_size )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dump_all_images( f'{output_path}/output', output_data_high_frequency )

    if dump_inputs:
        dump_all_images( f'{output_path}/input', input_data )

if __name__ == '__main__':
    diffuse_reflection_prediction_config = {
        'model_json_path': '../../data/diffuse_reflection.json',
        'model_weights_path': '../../data/diffuse_reflection.weights',
        'input_path': '../../data/wall_mirror_input.npy',
        'output_path': '../../data/diffuse_reconstruction_outputs',
        'learning_phase_fix': True,
        'batch_size': 1,
        'dump_inputs': True
    }
    make_prediction( diffuse_reflection_prediction_config )

