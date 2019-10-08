from keras.engine import Layer
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects

class PhaseNormalization(Layer):
    def __init__(self, **kwargs):
        super(PhaseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('PhaseNormalization must be called on a list of tensors (2): Got: ' + str(inputs))
        a = inputs[0]
        b = inputs[1]
        df = a-b
        ab = K.concatenate([a,b], axis=-1)
        ab = ( ab - K.mean(ab) ) / ( K.std(ab)+K.epsilon() )
        df = ( df - K.min(df) ) / ( K.max(df)-K.min(df)+K.epsilon() ) * 2.0 - 1.0
        return K.concatenate( [ab, df], axis=-1 )

    def compute_output_shape(self, input_shape):
        batch_size, row, col, channel = input_shape[0]
        return (batch_size, row, col, channel*3 )

get_custom_objects().update({'PhaseNormalization': PhaseNormalization})

if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    ip = Input(shape=(1, 1, 16))
    ix = Input(shape=(1, 1, 16))
    x = PhaseNormalization()([ip, ix])
    model = Model([ip,ix], x)
    model.summary()
