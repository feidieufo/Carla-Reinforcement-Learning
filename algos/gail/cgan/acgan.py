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


# def define_generator(latent_dim=50, nclasses=10):
#     label = layers.Input(shape=(1, ))
#     li = layers.Embedding(nclasses, latent_dim)(label)
#     li = layers.Flatten()(li)
#     noise = layers.Input(shape=(latent_dim, ))

#     input = layers.multiply([li, noise])
#     x = layers.Dense(7*7*128, activation="relu")(input)
#     x = layers.Reshape((7, 7, 128))(x)
#     x = layers.BatchNormalization(momentum=0.8)(x)
#     x = layers.UpSampling2D()(x)

#     x = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization(momentum=0.8)(x)
#     x = layers.UpSampling2D()(x)

#     x = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
#     x = layers.BatchNormalization(momentum=0.8)(x)

#     out = layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="tanh")(x)
#     model = tf.keras.Model([noise, label], out)

#     return model

# def define_discriminator(in_shape=(28, 28, 1), nclasses=10):
#     img = layers.Input(shape=in_shape)
#     x = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding="same")(img)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Dropout(0.25)(x)
#
#     x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
#     x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Dropout(0.25)(x)
#     x = layers.BatchNormalization(momentum=0.8)(x)
#
#     x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Dropout(0.25)(x)
#     x = layers.BatchNormalization(momentum=0.8)(x)
#
#     x = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Dropout(0.25)(x)
#
#     x = layers.Flatten()(x)
#
#     out = layers.Dense(1, activation="sigmoid")(x)
#     label = layers.Dense(nclasses, activation="softmax")(out)
#
#     model = tf.keras.Model(img, [out, label])
#     return model

def define_generator(latent_dim=50, nclasses=10):
    label = layers.Input(shape=(1, ))
    li = layers.Embedding(nclasses, 50)(label)
    li = layers.Dense(7*7*1, activation="relu")(li)
    li = layers.Reshape((7, 7, 1))(li)

    noise = layers.Input(shape=(latent_dim, ))
    n = layers.Dense(7*7*384, activation="relu")(noise)
    n = layers.Reshape((7, 7, 384))(n)

    input = layers.concatenate([n, li], axis=-1)
    x = layers.Conv2DTranspose(filters=192, kernel_size=5, strides=2, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding="same", activation="tanh")(x)

    model = tf.keras.Model([noise, label], x)

    return model


def define_discriminator(in_shape=(28, 28, 1), nclasses=10):
    img = layers.Input(shape=in_shape)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(img)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, 3, 2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    label = layers.Dense(nclasses, activation="softmax")(out)

    model = tf.keras.Model(img, [out, label])
    return model

@tf.function
def train_d(x, y, discriminator, true_out, fake_out, train_d_loss, fake_img, noise_label, opti_d):
    with tf.GradientTape() as tape:
        dis_out, dis_label = discriminator(x)
        d1_loss = tf.keras.losses.binary_crossentropy(true_out, dis_out) + \
                  tf.keras.losses.sparse_categorical_crossentropy(y, dis_label)

        dis2_out, dis2_label = discriminator(fake_img)
        d2_loss = tf.keras.losses.binary_crossentropy(fake_out, dis2_out) + \
                  tf.keras.losses.sparse_categorical_crossentropy(noise_label, dis2_label)

        d_loss = d1_loss + d2_loss
        train_d_loss(d_loss)

    d_gra = tape.gradient(d_loss, discriminator.trainable_variables)
    opti_d.apply_gradients(zip(d_gra, discriminator.trainable_variables))
    return d_gra

@tf.function
def train_g(generator, noise, noise_label, discriminator, true_out, train_g_loss, opti_g):
    with tf.GradientTape() as tape:
        gen_out = generator([noise, noise_label])
        dis_out, dis_label = discriminator(gen_out)

        g_loss = tf.keras.losses.binary_crossentropy(true_out, dis_out) + \
                 tf.keras.losses.sparse_categorical_crossentropy(noise_label, dis_label)
        train_g_loss(g_loss)

    g_gra = tape.gradient(g_loss, generator.trainable_variables)
    opti_g.apply_gradients(zip(g_gra, generator.trainable_variables))
    return g_gra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--l_dim", type=int, default=50)
    parser.add_argument("--b_size", type=int, default=100)
    parser.add_argument("--saver", type=str, default="saver_acgan")
    parser.add_argument("--logs", type=str, default="logs")
    parser.add_argument("--log_file", type=str, default="acgan")
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--update_d", type=int, default=5)
    parser.add_argument("--update_g", type=int, default=1)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    discriminator = define_discriminator()
    generator = define_generator(latent_dim=args.l_dim)
    discriminator.summary()
    generator.summary()

    # save_file = osp.join(osp.abspath(osp.dirname(__file__)), args.saver)
    # checkpoint = tf.train.Checkpoint(discriminator=discriminator, generator=generator)
    # summary_file = osp.join(osp.abspath(osp.dirname(__file__)), args.logs)
    # summary = tf.summary.create_file_writer(os.path.join(summary_file, args.log_file))
    #
    # (train_x, train_y), (test_x, test_y) = load_data()
    # train_x = (train_x - 127.5) / 127.5
    # test_x = (test_x - 127.5) / 127.5
    # train_x, test_x = np.expand_dims(train_x, axis=-1), np.expand_dims(test_x, axis=-1)
    # train_y, test_y = np.expand_dims(train_y, axis=-1), np.expand_dims(test_y, axis=-1)
    # train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(args.b_size)
    #
    # opti_d = tf.keras.optimizers.Adam(args.lr_d, 0.5)
    # opti_g = tf.keras.optimizers.Adam(args.lr_g, 0.5)
    #
    # train_d_loss = tf.keras.metrics.Mean(name='train_d_loss')
    # train_g_loss = tf.keras.metrics.Mean(name='train_g_loss')
    #
    # for itr in range(args.epoch):
    #     train_d_loss.reset_states()
    #     train_g_loss.reset_states()
    #
    #     for (x, y) in train_data:
    #         noise_label = np.random.randint(low=0, high=10, size=(args.b_size, 1))
    #         noise = np.random.randn(args.b_size, args.l_dim)
    #         fake_img = generator([noise, noise_label])
    #
    #         true_out = np.ones(shape=[args.b_size, 1])
    #         fake_out = np.zeros(shape=[args.b_size, 1])
    #         # fake_label = 10 * np.ones(shape=[args.b_size, 1])
    #
    #         for d_itr in range(args.update_d):
    #             d_gra = train_d(x, y, discriminator, true_out, fake_out, train_d_loss, fake_img, noise_label, opti_d)
    #
    #         for g_itr in range(args.update_g):
    #             g_gra = train_g(generator, noise, noise_label, discriminator, true_out, train_g_loss, opti_g)
    #
    #     print("itr", itr, "dloss", train_d_loss.result().numpy())
    #     print("itr", itr, "gloss", train_g_loss.result().numpy())
    #     with summary.as_default():
    #         tf.summary.scalar("dloss", train_d_loss.result(), itr)
    #         tf.summary.scalar("gloss", train_g_loss.result(), itr)
    #
    #         for (var, gra) in zip(discriminator.trainable_variables, d_gra):
    #             tf.summary.histogram("dis_" + var.name, gra, itr)
    #
    #         for (var, gra) in zip(generator.trainable_variables, g_gra):
    #             tf.summary.histogram("gen_" + var.name, gra, itr)
    #
    #     checkpoint.save(osp.join(save_file, "model.ckpt"))


if __name__ == '__main__':
    main()
























































































































































































































































































































































