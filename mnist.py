from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from model import GAN

IMAGE_DIR = Path("images")

IMG_ROWS = 28
IMG_COLUMNS = 28
IMG_CHANNELS = 1

BATCH_SIZE = 64
LATENT_DIM = 128
LEARNING_RATE = 0.0003
EPOCHS = 400


class ImageLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        latent_dim = self.model.latent_dim
        noise = tf.random.normal((25, latent_dim))
        generated_images = self.model(noise)
        generated_images = tf.squeeze(generated_images, axis=-1)

        with plt.ioff():
            plt.figure(figsize=(5, 5))
            for idx, image in enumerate(generated_images, start=1):
                plt.subplot(5, 5, idx)
                plt.imshow(image, cmap="gray", vmin=-1, vmax=1)
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(IMAGE_DIR / f"{epoch}.png")
            plt.close()


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 127.5 - 1.0
all_digits = np.expand_dims(all_digits, axis=-1)
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
buffer_size = len(dataset)
dataset = dataset.shuffle(buffer_size).batch(BATCH_SIZE)

gan = GAN(IMG_ROWS, IMG_COLUMNS, IMG_CHANNELS, LATENT_DIM)
gan.compile(
    d_optimizer=keras.optimizers.Adam(LEARNING_RATE),
    g_optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


if __name__ == "__main__":
    image_logger = ImageLogger()
    gan.fit(dataset, epochs=2, steps_per_epoch=1000, callbacks=[image_logger])
