import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from model import GAN

IMG_ROWS = 28
IMG_COLUMNS = 28
IMG_CHANNELS = 1
BATCH_SIZE = 64
LATENT_DIM = 128
LEARNING_RATE = 0.0003
EPOCHS = 1


class PNGLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        latent_dim = self.model.latent_dim
        noise = tf.random.normal((100, latent_dim))
        generated_images = self.model(noise)

        plt.figure(figsize=(10, 10))
        for i, image in enumerate(generated_images):
            plt.subplot(10, 10, i + 1)
            plt.imshow(image, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{epoch}.png")


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 127.5 - 1
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

gan = GAN(IMG_ROWS, IMG_COLUMNS, IMG_CHANNELS, LATENT_DIM)
gan.compile(
    d_optimizer=keras.optimizers.Adam(LEARNING_RATE),
    g_optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)


if __name__ == "__main__":
    png_logger = PNGLogger()
    gan.fit(dataset, epochs=EPOCHS, callbacks=[png_logger])
