import tensorflow as tf
import numpy as np
import random
import argparse
import os
import os.path as osp

try:
    import tensorflow.python.keras.layers as layers
    from tensorflow.python.keras.datasets.mnist import load_data
except Exception:
    import tensorflow.keras.layers as layers
    from tensorflow.keras.datasets.mnist import load_data


def define_generator(gen_input_size):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    in_lat = layers.Input(shape=(gen_input_size,))

    gen = layers.Dense(512*7*7, kernel_initializer=init, activation="relu")(in_lat)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Reshape((7, 7, 512))(gen)

    gen = layers.Conv2D(128, kernel_size=4, padding='same', kernel_initializer=init, activation="relu")(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same",
                                 kernel_initializer=init, activation="relu")(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same",
                                 kernel_initializer=init, activation="tanh")(gen)

    model = tf.keras.Model(in_lat, gen)
    return model

def define_discriminator(in_shape=(28, 28, 1), nclasses=10):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    in_image = layers.Input(shape=in_shape)

    x = layers.Conv2D(64, 4, 2, padding="same", kernel_initializer=init)(in_image)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(128, 4, 2, padding="same", kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 4, padding="same", kernel_initializer=init)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    d = layers.Dense(1, activation="sigmoid")(x)

    d_model = tf.keras.Model(in_image, d)

    q = layers.Dense(128)(x)
    q = layers.BatchNormalization()(q)
    q = layers.LeakyReLU(alpha=0.1)(q)
    q = layers.Dense(nclasses, activation="softmax")(q)

    q_model = tf.keras.Model(in_image, q)

    return d_model, q_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--l_dim", type=int, default=50)
    parser.add_argument("--nclasses", type=int, default=10)
    parser.add_argument("--b_size", type=int, default=100)
    parser.add_argument("--saver", type=str, default="saver_infogan")
    parser.add_argument("--logs", type=str, default="logs")
    parser.add_argument("--log_file", type=str, default="infogan")
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    d_model, q_model = define_discriminator()
    g_model = define_generator(gen_input_size=args.l_dim+args.nclasses)
    d_model.summary()
    q_model.summary()
    g_model.summary()

    save_file = osp.join(osp.abspath(osp.dirname(__file__)), args.saver)
    checkpoint = tf.train.Checkpoint(d_model=d_model, g_model=g_model, q_model=q_model)
    summary_file = osp.join(osp.abspath(osp.dirname(__file__)), args.logs)
    summary = tf.summary.create_file_writer(os.path.join(summary_file, args.log_file))

    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = (train_x - 127.5) / 127.5
    test_x = (test_x - 127.5) / 127.5
    train_x, test_x = np.expand_dims(train_x, axis=-1), np.expand_dims(test_x, axis=-1)
    train_y, test_y = np.expand_dims(train_y, axis=-1), np.expand_dims(test_y, axis=-1)
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(args.b_size)

    opti_d = tf.keras.optimizers.Adam(args.lr_d, 0.5)
    opti_g = tf.keras.optimizers.Adam(args.lr_g, 0.5)

    train_d_loss = tf.keras.metrics.Mean(name='train_d_loss')
    train_g_loss = tf.keras.metrics.Mean(name='train_g_loss')

    for itr in range(args.epoch):
        train_d_loss.reset_states()
        train_g_loss.reset_states()

        for (x, y) in train_data:
            z = np.random.randn(args.b_size, args.l_dim)
            c = np.random.randint(0, 10, size=(args.b_size, ))
            c = tf.keras.utils.to_categorical(c, num_classes=args.nclasses)
            in_lat = np.hstack((z, c))

            fake_imgs = g_model(in_lat)
            true_out = np.ones((args.b_size, 1))
            fake_out = np.zeros((args.b_size, 1))
            d_model.trainable = True
            with tf.GradientTape() as tape:
                dis1_out = d_model(x)
                d1_loss = tf.keras.losses.binary_crossentropy(true_out, dis1_out)
                dis2_out = d_model(fake_imgs)
                d2_loss = tf.keras.losses.binary_crossentropy(fake_out, dis2_out)

                d_loss = d1_loss + d2_loss
                train_d_loss(d_loss)

            d_gra = tape.gradient(d_loss, d_model.trainable_variables)
            opti_d.apply_gradients(zip(d_gra, d_model.trainable_variables))

            d_model.trainable = False
            with tf.GradientTape() as tape:
                fake = g_model(in_lat)
                dis_out = d_model(fake)
                q_out = q_model(fake)

                g_loss = tf.keras.losses.binary_crossentropy(true_out, dis_out) + \
                         tf.keras.losses.categorical_crossentropy(c, q_out)
                train_g_loss(g_loss)

            var = g_model.trainable_variables + q_model.trainable_variables
            g_gra = tape.gradient(g_loss, var)
            opti_g.apply_gradients(zip(g_gra, var))

        print("itr", itr, "dloss", train_d_loss.result().numpy())
        print("itr", itr, "gloss", train_g_loss.result().numpy())
        with summary.as_default():
            tf.summary.scalar("dloss", train_d_loss.result(), itr)
            tf.summary.scalar("gloss", train_g_loss.result(), itr)

        checkpoint.save(osp.join(save_file, "model.ckpt"))






