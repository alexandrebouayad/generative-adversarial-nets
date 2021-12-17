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
        self.combined = Sequential([self.generator, self.discriminator])

    def _build_discriminator(self):
        return Sequential(
            [
                # layers.InputLayer(self.img_shape),
                layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPool2D(),
                layers.Dense(units=1),
            ]
        )

    def _build_generator(self):
        return Sequential(
            [
                # layers.InputLayer((self.latent_dim,)),
                layers.Dense(7 * 7 * 128),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(
                    filters=1,
                    kernel_size=(7, 7),
                    padding="same",
                    activation="sigmoid",
                ),
            ]
        )

    def call(self, inputs):
        return self.generator(inputs)

    def compile(self, d_optimizer, g_optimizer, loss):
        super().compile()
        self.discriminator.compile(d_optimizer, loss)
        self.combined.compile(g_optimizer, loss)

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        latent_shape = batch_size, self.latent_dim

        noise = tf.random.normal(latent_shape)
        generated_images = self.generator(noise)
        images = tf.concat([real_images, generated_images], axis=0)
        zeros = tf.zeros((batch_size, 1))
        ones = tf.ones((batch_size, 1))
        labels = tf.concat([zeros, ones], axis=0)
        d_loss = self.discriminator.train_step((images, labels))
        d_loss["d_loss"] = d_loss.pop("loss")

        noise = tf.random.normal(latent_shape)
        labels = tf.zeros((batch_size, 1))
        self.discriminator.trainable = False
        g_loss = self.combined.train_step((noise, labels))
        self.discriminator.trainable = True
        g_loss["g_loss"] = g_loss.pop("loss")

        return d_loss | g_loss
