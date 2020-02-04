import tensorflow as tf
import numpy as np
import os.path as osp
import os
import argparse
try:
    import tensorflow.python.keras.layers as layers
    from tensorflow.python.keras.datasets.mnist import load_data
except Exception:
    import tensorflow.keras.layers as layers
    from tensorflow.keras.datasets.mnist import load_data


def define_generator(latent_dim=50, nclasses=10):
    label = layers.Input(shape=(1, ))
    li = layers.Embedding(nclasses, latent_dim)(label)
    li = layers.Flatten()(li)
    noise = layers.Input(shape=(latent_dim, ))

    input = layers.multiply([li, noise])
    x = layers.Dense(7*7*128, activation="relu")(input)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.UpSampling2D()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    out = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="tanh")(x)
    model = tf.keras.Model([noise, label], out)

    return model


def define_discriminator(in_shape=(28, 28, 1), nclasses=10):
    img = layers.Input(shape=in_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding="same")(img)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    label = layers.Dense(nclasses+1, activation="softmax")(out)

    model = tf.keras.Model(img, [out, label])
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saver", type=str, default="saver_kerasloss_ldim100")
    parser.add_argument("--l_dim", type=int, default=100)
    parser.add_argument("--b_size", type=int, default=10)
    parser.add_argument("--filename", type=str, default="model.ckpt-50")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    discriminator = define_discriminator()
    generator = define_generator(latent_dim=args.l_dim)

    save_file = osp.join(osp.abspath(osp.dirname(__file__)), args.saver)
    checkpoint = tf.train.Checkpoint(discriminator=discriminator, generator=generator)
    checkpoint.restore(os.path.join(save_file, args.filename))

    summary_file = osp.join(osp.abspath(osp.dirname(__file__)), "logs")
    summary = tf.summary.create_file_writer(osp.join(summary_file, "acgan_image" + args.saver[5:]))

    latent = np.random.normal(0, 1, size=(args.b_size, args.l_dim))
    for i in range(10):
        label = np.expand_dims(np.array([i]*args.b_size), 1)
        image = generator([latent, label])

        with summary.as_default():
            tf.summary.image("gen"+str(i), image, max_outputs=args.b_size, step=0)
