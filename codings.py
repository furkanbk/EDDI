import tensorflow as tf

def fc_uci_decoder(z, obs_dim, activation='sigmoid'):  # Output means, model is N(m,sigmaI) or Bernoulli(m)
    x = tf.keras.layers.Dense(50, activation=None, name='fc_01')(z)
    x = tf.keras.layers.Dense(100, activation=None, name='fc_02')(x)
    x = tf.keras.layers.Dense(obs_dim, activation=activation, name='fc_final')(x)
    return x, None

def fc_uci_encoder(x, latent_dim, activation=None):
    e = tf.keras.layers.Dense(100, activation=None, name='fc_01')(x)
    e = tf.keras.layers.Dense(50, activation=None, name='fc_02')(e)
    e = tf.keras.layers.Dense(2 * latent_dim, activation=activation, name='fc_final')(e)
    return e

def PNP_fc_uci_encoder(x, K, activation=None):
    e = tf.keras.layers.Dense(100, activation=None, name='fc_01')(x)
    e = tf.keras.layers.Dense(50, activation=None, name='fc_02')(e)
    e = tf.keras.layers.Dense(K, activation=activation, name='fc_final')(e)
    return e
