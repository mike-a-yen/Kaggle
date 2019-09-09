import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def build_model(model_params, **kwargs):
    input_layer = layers.Input(shape=(None,))
    encoder = Encoder(model_params, name='encoder')(input_layer)
    logit = layers.Dense(1)(encoder)
    act = layers.Activation('sigmoid')(logit)
    model = models.Model(input_layer, act)
    return model


def compile_model(model) -> None:
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
    )
    return

class Encoder(models.Model):
    def __init__(self, model_params: dict, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.embedding_layer = layers.Embedding(
            model_params['vocab_size'],
            model_params['embedding_size']
        )

        self.rnn_cell = layers.LSTM(
            model_params['output_size']
        )

    def call(self, X):
        embs = self.embedding_layer(X)
        embs = self.rnn_cell(embs)
        return embs
