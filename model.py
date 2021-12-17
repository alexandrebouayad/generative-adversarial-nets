import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers


class GAN(Model):
    def __init__(self, img_rows, ing_columns, img_channels, latent_dim):
        super().__init__()
        self.img_shape = img_rows, ing_columns, img_channels
        self.latent_dim = latent_dim
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        # self.discriminator.trainable = False
        self.combined = Sequential([self.generator, self.discriminator])

    def _build_discriminator(self):
        return Sequential(
            [
                layers.Flatten(input_shape=self.img_shape),
                layers.Dense(units=512),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(units=256),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(units=1),
            ]
        )

    def _build_generator(self):
        return Sequential(
            [
                layers.Dense(units=256, input_dim=self.latent_dim),
                layers.LeakyReLU(alpha=0.2),
                layers.BatchNormalization(momentum=0.8),
                layers.Dense(units=512),
                layers.LeakyReLU(alpha=0.2),
                layers.BatchNormalization(momentum=0.8),
                layers.Dense(units=1024),
                layers.LeakyReLU(alpha=0.2),
                layers.BatchNormalization(momentum=0.8),
                layers.Dense(np.prod(self.img_shape), activation="tanh"),
                layers.Reshape(self.img_shape),
            ]
        )

    def call(self, inputs):
        return self.generator(inputs)

    def compile(self, d_optimizer, g_optimizer, loss, metrics):
        super().compile()
        self.discriminator.compile(d_optimizer, loss, metrics)
        self.combined.compile(g_optimizer, loss, metrics)

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        latent_shape = batch_size, self.latent_dim

        ones = tf.ones((batch_size, 1))
        zeros = tf.zeros((batch_size, 1))
        ones_zeros = tf.concat([ones, zeros], axis=0)

        noise = tf.random.normal(latent_shape)
        generated_images = self.generator(noise)
        images = tf.concat([real_images, generated_images], axis=0)
        d_results = self.discriminator.train_step((images, ones_zeros))

        noise = tf.random.normal(latent_shape)
        self.discriminator.trainable = False
        g_results = self.combined.train_step((noise, ones))
        self.discriminator.trainable = True

        return {
            **{f"d_{key}": value for key, value in d_results.items()},
            **{f"g_{key}": value for key, value in g_results.items()},
        }
