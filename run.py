import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from model import GenerativeAdversarialNet

BATCH_SIZE = 64
LATENT_DIM = 128
LEARNING_RATE = 0.0003


def main():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    gan = GenerativeAdversarialNet(LATENT_DIM)
    gan.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    gan.fit(dataset)


if __name__ == "__main__":
    main()
