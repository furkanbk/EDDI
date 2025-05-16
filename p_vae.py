import tensorflow as tf
import numpy as np
import copy

class PN_Plus_VAE(tf.keras.Model):
    def __init__(self,
                 encoder,
                 decoder,
                 obs_dim,
                 learning_rate=1e-3,
                 obs_distrib="Gaussian",
                 obs_std=0.1 * np.sqrt(2),
                 K=20,
                 latent_dim=10,
                 batch_size=100,
                 M=5,
                 **kwargs):
        super(PN_Plus_VAE, self).__init__(**kwargs)
        
        self._K = K
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._obs_dim = obs_dim
        self._learning_rate = learning_rate
        self._obs_distrib = obs_distrib
        self._obs_std = obs_std
        self._M = M
        
        # Parameters F and b as tf.Variables, initialized randomly
        self.F = tf.Variable(
            initial_value=tf.random.normal([1, obs_dim, 10], stddev=0.1),
            trainable=True, name="F")
        self.b = tf.Variable(
            initial_value=tf.random.normal([1, obs_dim, 1], stddev=0.1),
            trainable=True, name="b")
        
        # Build optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
    
    def encode(self, x, mask):
        batch_size = tf.shape(x)[0]
        
        # Tile F and b for batch
        F_tiled = tf.tile(self.F, [batch_size, 1, 1])  # [batch_size, obs_dim, 10]
        b_tiled = tf.tile(self.b, [batch_size, 1, 1])  # [batch_size, obs_dim, 1]
        
        x_flat = tf.reshape(x, [-1, 1])  # [batch_size * obs_dim, 1]
        F_flat = tf.reshape(F_tiled, [-1, 10])  # [batch_size * obs_dim, 10]
        b_flat = tf.reshape(b_tiled, [-1, 1])  # [batch_size * obs_dim, 1]
        
        x_aug = tf.concat([x_flat, x_flat * F_flat, b_flat], axis=1)  # [batch_size * obs_dim, 1+10+1=12]
        
        # Fully connected layer 1
        encoded = tf.keras.layers.Dense(self._K, activation=None)(x_aug)  # [batch_size * obs_dim, K]
        encoded = tf.reshape(encoded, [batch_size, self._obs_dim, self._K])
        
        mask_expanded = tf.expand_dims(mask, axis=-1)  # [batch_size, obs_dim, 1]
        mask_tiled = tf.tile(mask_expanded, [1, 1, self._K])  # [batch_size, obs_dim, K]
        
        # Apply mask and sum over obs_dim
        encoded = tf.nn.relu(tf.reduce_sum(encoded * mask_tiled, axis=1))  # [batch_size, K]
        
        # Dense layers after aggregation
        encoded = tf.keras.layers.Dense(500, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(200, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(2 * self._latent_dim, activation=None)(encoded)  # mean and logvar
        
        mean = encoded[:, :self._latent_dim]
        logvar = encoded[:, self._latent_dim:]
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        stddev = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * stddev, stddev
    
    def decode(self, z):
        decoded, _ = self._decode(z, self._obs_dim)
        return decoded
    
    def compute_loss(self, x, mask):
        mean, logvar = self.encode(x, mask)
        z, stddev = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        
        # Reconstruction loss
        if self._obs_distrib == 'Bernoulli':
            eps = 1e-8
            rec_loss = -tf.reduce_sum(
                mask * (x * tf.math.log(decoded + eps) + (1 - x) * tf.math.log(1 - decoded + eps)), axis=1)
        else:
            # Gaussian negative log likelihood (ignoring constants)
            rec_loss = tf.reduce_sum(
                0.5 * tf.square((x - decoded) * mask) / (self._obs_std ** 2) + mask * tf.math.log(self._obs_std),
                axis=1)
        
        total_loss = tf.reduce_mean(kl_loss + rec_loss)
        return total_loss, tf.reduce_mean(kl_loss), tf.reduce_mean(rec_loss)
    
    @tf.function
    def train_step(self, x, mask):
        with tf.GradientTape() as tape:
            loss, kl, rec = self.compute_loss(x, mask)
        gradients = tape.gradient(loss, self.trainable_variables + [self.F, self.b])
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables + [self.F, self.b]))
        return loss, kl, rec
    
    def impute(self, x, mask_obs):
        imputed_samples = []
        for _ in range(self._M):
            mean, logvar = self.encode(x, mask_obs)
            z, _ = self.reparameterize(mean, logvar)
            decoded = self.decode(z)
            imputed_samples.append(decoded)
        return tf.reduce_mean(tf.stack(imputed_samples), axis=0)
    
    # You can add more methods similarly, like predictive_loss, update, etc., rewritten in TF2 style.


# Example usage:
# You need to define encoder and decoder functions compatible with this interface.
# Here, decoder(z, obs_dim) should return (decoded_tensor, some_other_output)

# Then you can instantiate and train like:

# vae = PN_Plus_VAE(encoder=your_encoder, decoder=your_decoder, obs_dim=your_obs_dim)
# for epoch in range(num_epochs):
#     loss, kl, rec = vae.train_step(x_batch, mask_batch)
