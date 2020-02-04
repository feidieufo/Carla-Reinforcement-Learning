import tensorflow as tf
import numpy as np
import random
import argparse
import os
import os.path as osp
try:
    import tensorflow.python.keras.layers as layers
    from tensorflow.python.keras.datasets.fashion_mnist import load_data
except Exception:
    import tensorflow.keras.layers as layers
    from tensorflow.keras.datasets.fashion_mnist import load_data


def define_discriminator(in_shape=(28, 28, 1), nclasses=10):
    label = layers.Input(shape=(1, ))
    li = layers.Embedding(nclasses, 50)(label)
    li = layers.Dense(in_shape[0]*in_shape[1])(li)
    li = layers.Reshape(in_shape)(li)
    image = layers.Input(shape=in_shape)
    x = layers.concatenate([image, li], axis=-1)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([image, label], out)
    return model


def define_generator(latent_dim=50, nclasses=10):
    label = layers.Input(shape=(1, ))
    li = layers.Embedding(nclasses, 50)(label)
    li = layers.Dense(7*7)(li)
    li = layers.Reshape((7, 7, 1))(li)

    in_lat = layers.Input((latent_dim,))
    lat = layers.Dense(7*7*128)(in_lat)
    lat = layers.Reshape((7, 7, 128))(lat)

    x = layers.concatenate([li, lat], axis=-1)
    x = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    out = layers.Conv2D(filters=1, kernel_size=7, activation="tanh", padding="same")(x)
    model = tf.keras.Model([label, in_lat], out)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--l_dim", type=int, default=50)
    parser.add_argument("--b_size", type=int, default=100)
    parser.add_argument("--saver", type=str, default="saver")
    parser.add_argument("--logs", type=str, default="logs")
    parser.add_argument("--log_file", type=str, default="cgan")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    discriminator = define_discriminator()
    generator = define_generator(latent_dim=args.l_dim)
    discriminator.summary()
    generator.summary()

    save_file = osp.join(osp.abspath(osp.dirname(__file__)), args.saver)
    checkpoint = tf.train.Checkpoint(discriminator=discriminator, generator=generator)


    opti_d = tf.keras.optimizers.Adam(0.0001)
    opti_g = tf.keras.optimizers.Adam(0.0001)
    summary_file = osp.join(osp.abspath(osp.dirname(__file__)), args.logs)
    summary = tf.summary.create_file_writer(os.path.join(summary_file, args.log_file))

    latent_dim = args.l_dim
    batch_size = args.b_size
    epochs = args.epoch

    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = (train_x - 127.0) / 127.0
    test_x = (test_x - 127.0) / 127.0
    train_x, test_x = np.expand_dims(train_x, axis=-1), np.expand_dims(test_x, axis=-1)
    train_y, test_y = np.expand_dims(train_y, axis=-1), np.expand_dims(test_y, axis=-1)
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(batch_size)

    train_d_loss = tf.keras.metrics.Mean(name='train_d_loss')
    train_g_loss = tf.keras.metrics.Mean(name='train_g_loss')

    for itr in range(epochs):
        train_d_loss.reset_states()
        train_g_loss.reset_states()

        for (x, y) in train_data:

            latent = np.random.normal(0, 1, size=(batch_size, latent_dim))
            label = np.random.randint(0, 9, size=(batch_size, 1))
            lake = generator([label, latent])
            with tf.GradientTape() as tape:
                dis1 = discriminator([x, y])
                dis2 = discriminator([lake, label])

                d1_loss = tf.keras.losses.binary_crossentropy(np.ones(shape=(batch_size, 1)), dis1)
                d2_loss = tf.keras.losses.binary_crossentropy(np.zeros(shape=(batch_size, 1)), dis2)
                dloss = d1_loss + d2_loss

                # dloss = tf.reduce_mean(tf.math.log(dis1)) + tf.reduce_mean(tf.math.log(1-dis2))
                # dloss = -dloss
                train_d_loss(dloss)

            gra_d = tape.gradient(dloss, discriminator.trainable_variables)
            opti_d.apply_gradients(zip(gra_d, discriminator.trainable_variables))

            latent = np.random.normal(0, 1, size=(batch_size, latent_dim))
            label = np.random.randint(0, 10, size=(batch_size, 1))

            with tf.GradientTape() as tape:
                gen = generator([label, latent])
                dis = discriminator([gen, label])

                # gloss = -tf.reduce_mean(tf.math.log(dis))
                gloss = tf.keras.losses.binary_crossentropy(np.ones(shape=(batch_size, 1)), dis)
                train_g_loss(gloss)

            gra_g = tape.gradient(gloss, generator.trainable_variables)
            opti_g.apply_gradients(zip(gra_g, generator.trainable_variables))

            # print("itr", itr, "dloss", dloss)
            # print("itr", itr, "gloss", gloss)

        print("itr", itr, "dloss", train_d_loss.result().numpy())
        print("itr", itr, "gloss", train_g_loss.result().numpy())
        with summary.as_default():
            tf.summary.scalar("dloss", train_d_loss.result(), itr)
            tf.summary.scalar("gloss", train_g_loss.result(), itr)

            for (var, gra) in zip(discriminator.trainable_variables, gra_d):
                tf.summary.histogram("dis" + var.name, gra, itr)

            for (var, gra) in zip(generator.trainable_variables, gra_g):
                tf.summary.histogram("gen" + var.name, gra, itr)


        checkpoint.save(osp.join(save_file, "model.ckpt"))

    #
    # print('Train', train_x.shape, train_y.shape)
    # print('Test', test_x.shape, test_y.shape)