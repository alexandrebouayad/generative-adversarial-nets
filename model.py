import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential, layers


class GenerativeAdversarialNet(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.discriminator = Sequential(
            [
                Input(shape=(28, 28, 1)),
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
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )
        self.generator = Sequential(
            [
                Input(shape=(latent_dim,)),
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
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        self.latent_dim = latent_dim

    def train_step(self, real_images):
        # real_images, _ = data
        batch_size = tf.shape(real_images)[0]
        latent_input_shape = batch_size, self.latent_dim

        random_latent_vectors = tf.random.normal(latent_input_shape)
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
            axis=0,
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.compiled_loss(
                labels,
                predictions,
                regularization_losses=self.losses,
            )

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        grads_and_vars = zip(grads, self.discriminator.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)

        random_latent_vectors = tf.random.normal(latent_input_shape)
        fooling_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(generated_images)
            g_loss = self.compiled_loss(
                fooling_labels,
                predictions,
                regularization_losses=self.losses,
            )

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        grads_and_vars = zip(grads, self.generator.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)

        return {"d_loss": d_loss, "g_loss": g_loss}
