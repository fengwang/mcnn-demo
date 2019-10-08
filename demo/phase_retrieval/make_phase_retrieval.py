import numpy as np
a = np.load( '../../data/lhs_16.npy' ) # lhs experimental data, 16 channels
b = np.load( '../../data/rhs_16.npy' ) # rhs experimental data, 16 channels

from phase_norm import PhaseNormalization
from keras.models import load_model
from group_norm import GroupNormalization

mm = load_model( '../../data/phase_retrieval_model_32.model' )
result = mm.predict( [a, b] )

import imageio # save predicted phase and amplitude
imageio.imsave( '../../data/p_amplitude.png', np.squeeze( result[:,100:540,60:580,0] ) )#ROI
imageio.imsave( '../../data/p_phase.png', np.squeeze( result[:,100:540,60:580,1] ) )#ROI

a = a / np.amax(a)
b = b / np.amax(b)

for idx in range( 16 ): # save experimental data
    imageio.imsave( f'../../data/exp_lhs_{idx}.png', np.squeeze(a)[:,:,idx] )
    imageio.imsave( f'../../data/exp_rhs_{idx}.png', np.squeeze(b)[:,:,idx] )

