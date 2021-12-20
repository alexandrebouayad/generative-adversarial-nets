import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from model import GAN

IMAGE_DIR = Path("images/")
CHECKPOINT_FILE = Path("checkpoints/model/")

IMG_ROWS = 28
IMG_COLUMNS = 28
IMG_CHANNELS = 1

LATENT_DIM = 128
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
EPOCHS = 250


def build_dataset(batch_size):
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    real_images = np.concatenate([x_train, x_test])
    real_images = real_images.astype("float32") / 127.5 - 1.0
    real_images = np.expand_dims(real_images, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(real_images)
    buffer_size = len(dataset)

    return dataset.shuffle(buffer_size).batch(batch_size)


class ImageLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        latent_dim = self.model.latent_dim
        noise = tf.random.normal((25, latent_dim))
        generated_images = self.model(noise)
        generated_images = tf.squeeze(generated_images, axis=-1)

        plt.figure(figsize=(5, 5))
        for idx, image in enumerate(generated_images, start=1):
            plt.subplot(5, 5, idx)
            plt.imshow(image, cmap="gray", vmin=-1, vmax=1)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"{epoch}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--d-learning-rate", type=int, default=LEARNING_RATE)
    parser.add_argument("--g-learning-rate", type=int, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    gan = GAN(IMG_ROWS, IMG_COLUMNS, IMG_CHANNELS, args.latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(args.d_learning_rate),
        g_optimizer=keras.optimizers.Adam(args.g_learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    dataset = build_dataset(args.batch_size)
    callbacks = [ImageLogger(), ModelCheckpoint(args.checkpoint)]

    gan.fit(dataset, epochs=args.epochs, callbacks=callbacks)
