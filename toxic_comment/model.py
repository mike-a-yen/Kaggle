import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers


def build_model(model_params, **kwargs):
    input_layer = layers.Input(shape=(None,))
    encoder = Encoder(model_params, name='encoder')(input_layer)
    logit = layers.Dense(model_params['targets'])(encoder)
    act = layers.Activation('sigmoid')(logit)
    model = models.Model(input_layer, act)
    return model


def compile_model(model, model_params) -> None:
    opt_params = model_params['opt_params']
    opt = optimizers.Adam(
        **opt_params
    )
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']
    )
    return


class HighwayBlock(layers.Layer):
    def __init__(self, input_size: int, **kwargs) -> None:
        super(HighwayBlock, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_size = input_size

    def build(self, input_shape) -> None:
        self.W = self.add_weight(
            name='W',
            shape=(self.input_size, self.output_size),
            initializer='he_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.output_size,),
            initializer='zeros',
            trainable=True
        )
        self.Wt = self.add_weight(
            name='W_t',
            shape=(self.input_size, self.output_size),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bt = self.add_weight(
            name='bias_t',
            shape=(self.output_size,),
            initializer='zeros',
            trainable=True
        )
        super(HighwayBlock, self).build(input_shape)

    def call(self, X):
        transform = tf.sigmoid(tf.matmul(X, self.Wt) + self.bt)
        carry = 1 - transform
        g = tf.nn.relu(tf.matmul(X, self.W) + self.b)
        return g * transform + X * carry

    def get_config(self) -> dict:
        base_config = super(Encoder, self).get_config()
        config = {
            'input_size': self.input_size
        }
        config.update(base_config)
        return config


class Encoder(models.Model):
    def __init__(self, model_params: dict, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.params = model_params

        self.embedding_layer = layers.Embedding(
            model_params['vocab_size'],
            model_params['embedding_size']
        )
        self.conv_layer = layers.Conv1D(
            model_params['num_kernels'],
            model_params['width'],
            padding='valid'
        )
        self.highway_layer = HighwayBlock(model_params['num_kernels'])
        self.pool_layer = layers.MaxPooling1D(model_params['width'])
        self.rnn_cell = layers.CuDNNLSTM(
            model_params['output_size']
        )

    def call(self, X):
        embs = self.embedding_layer(X)
        embs = self.conv_layer(embs)
        embs = self.highway_layer(embs)
        embs = self.pool_layer(embs)
        embs = self.rnn_cell(embs)
        return embs

    def get_config(self) -> dict:
        config = {
            'model_params': self.params
        }
        return config
