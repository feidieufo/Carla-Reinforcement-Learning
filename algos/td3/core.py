import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers


# class CnnBlock(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = layers.Conv2D(32, (8, 8), strides=4, activation="relu", name="conv1")
#         self.conv2 = layers.Conv2D(64, (4, 4), strides=2, activation="relu", name="conv2")
#         self.conv3 = layers.Conv2D(64, (3, 3), strides=1, activation="relu", name="conv3")
#         self.dense = layers.Dense(512, activation="relu")
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv1(inputs)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = layers.Flatten()(x)
#         x = self.dense(x)
#         return x
#
#
# class FCBlock(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = layers.Dense(units=100, activation="relu")
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.dense(inputs)
#         return x
#
class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(128, activation="relu")

        self.dense1 = [layers.Dense(256, activation="relu") for i in range(8)]
        self.dense2 = [layers.Dense(1) for i in range(4)]

    def call(self, inputs, training=None, mask=None):
        image = inputs[0]
        action = self.dense(inputs[1])
        x = layers.concatenate([image, action], axis=1)
        branches = []
        for i in range(4):
            branch = self.dense1[i*2](x)
            branch = self.dense1[i*2+1](branch)
            branch = self.dense2[i](branch)

            branches.append(branch)
        return branches
#
#
# class Actor(tf.keras.Model):
#     def __init__(self, act_dim, act_limit):
#         super().__init__()
#         self.act_limit = act_limit
#         self.dense1 = layers.Dense(256, activation="relu")
#         self.dense2 = layers.Dense(256, activation="relu")
#         self.dense3 = layers.Dense(act_dim, activation="tanh")
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         x = self.dense3(x)
#         x = x*self.act_limit
#         return x
#
#
# class ActorCritic(tf.keras.Model):
#     def __init__(self, act_dim, act_limit):
#         super().__init__()
#         self.cnn = CnnBlock()
#         # self.cnn = FCBlock()
#         self.q1 = Critic()
#         self.q2 = Critic()
#         self.pi = Actor(act_dim, act_limit)
#
#     def call(self, inputs, training=None, mask=None):
#         image = self.cnn(inputs[0])
#         q1 = self.q1([image, inputs[1]])
#         q2 = self.q2([image, inputs[1]])
#         return q1, q2
#
#     def choose_action(self, inputs):
#         x = self.cnn(inputs)
#         x = self.pi(x)
#
#         return x


class ActorCritic(tf.keras.Model):
    def __init__(self, act_dim, act_limit):
        super().__init__()
        self.q1 = Critic()
        self.q2 = Critic()
        self.actor = ActorCnn_()

    def call(self, inputs, training=False, mask=None):
        image = inputs[0]
        speed = inputs[1]
        action = inputs[2]
        x, z = self.actor.emd([image, speed], training=training)
        q1 = self.q1([z, action], training=training)
        q2 = self.q2([z, action], training=training)

        return q1, q2

    def select_action(self, inputs, training=False):
        branch, z = self.actor(inputs, training=training)
        return branch, z

    def work_q1(self, z, action, training=False):
        q1 = self.q1([z, action])
        return q1


class ConvBnDropout(tf.keras.Model):
    def __init__(self, filters, strides, kernel_size, padding="valid", rate=0.2):
        super().__init__(trainable=False)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(1 - rate)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inpus, training=False):
        x = self.conv1(inpus)
        x = self.bn(x, training=training)
        if training:
            x = self.drop(x)
        x = self.relu(x)
        return x


class FCDropout(tf.keras.Model):
    def __init__(self, units, rate=0.4):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=units)
        self.drop = tf.keras.layers.Dropout(1 - rate)
        self.relu = tf.keras.layers.ReLU()
        # self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=units),
        #                         tf.keras.layers.Dropout(rate),
        #                         tf.keras.layers.ReLU()])

    def call(self, inpus, training=False):
        x = self.dense(inpus)
        if training:
            x = self.drop(x)
        x = self.relu(x)
        return x


class ActorEmd(tf.keras.Model):
    def __init__(self, dropout=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5):
        super().__init__()
        self.conv1 = ConvBnDropout(filters=32, strides=2, kernel_size=5, padding="valid", rate=dropout[0])
        self.conv2 = ConvBnDropout(filters=32, strides=1, kernel_size=3, padding="valid", rate=dropout[1])
        self.conv3 = ConvBnDropout(filters=64, strides=2, kernel_size=3, padding="valid", rate=dropout[2])
        self.conv4 = ConvBnDropout(filters=64, strides=1, kernel_size=3, padding="valid", rate=dropout[3])
        self.conv5 = ConvBnDropout(filters=128, strides=2, kernel_size=3, padding="valid", rate=dropout[4])
        self.conv6 = ConvBnDropout(filters=128, strides=1, kernel_size=3, padding="valid", rate=dropout[5])
        self.conv7 = ConvBnDropout(filters=256, strides=1, kernel_size=3, padding="valid", rate=dropout[6])
        self.conv8 = ConvBnDropout(filters=256, strides=1, kernel_size=3, padding="valid", rate=dropout[7])

        self.flat = tf.keras.layers.Flatten()
        self.dense1 = FCDropout(units=512, rate=dropout[8])
        self.dense2 = FCDropout(units=512, rate=dropout[9])

        self.dense3 = FCDropout(units=128, rate=dropout[10])
        self.dense4 = FCDropout(units=128, rate=dropout[11])

        self.dense5 = FCDropout(units=512, rate=dropout[12])

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs[0], training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        x = self.conv8(x, training=training)

        x = self.flat(x)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        y = self.dense3(inputs[1], training=training)
        y = self.dense4(y, training=training)

        z = tf.keras.layers.concatenate([x, y], axis=1)
        z = self.dense5(z, training=training)

        return x, z


class ActorCnn_(tf.keras.Model):
    def __init__(self, dropout=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5):
        super().__init__()
        self.emd = ActorEmd(dropout)

        self.dense = [FCDropout(units=256, rate=dropout[13 + i]) for i in range(10)]

        self.out = [tf.keras.layers.Dense(3) for i in range(4)]
        self.out.append(tf.keras.layers.Dense(1))

    def call(self, inputs, training=False, mask=None):

        x, z = self.emd(inputs, training=training)

        branchs = []
        for i in range(4):
            branch = self.dense[i * 2](z, training=training)
            branch = self.dense[i * 2 + 1](branch, training=training)

            branchs.append(self.out[i](branch))

        branch = self.dense[8](x, training=training)
        branch = self.dense[9](branch, training=training)
        branchs.append(self.out[4](branch))

        return branchs, z


class ActorCnn(tf.keras.Model):
    def __init__(self, dropout=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5):
        super().__init__()
        self.conv1 = ConvBnDropout(filters=32, strides=2, kernel_size=5, padding="valid", rate=dropout[0])
        self.conv2 = ConvBnDropout(filters=32, strides=1, kernel_size=3, padding="valid", rate=dropout[1])
        self.conv3 = ConvBnDropout(filters=64, strides=2, kernel_size=3, padding="valid", rate=dropout[2])
        self.conv4 = ConvBnDropout(filters=64, strides=1, kernel_size=3, padding="valid", rate=dropout[3])
        self.conv5 = ConvBnDropout(filters=128, strides=2, kernel_size=3, padding="valid", rate=dropout[4])
        self.conv6 = ConvBnDropout(filters=128, strides=1, kernel_size=3, padding="valid", rate=dropout[5])
        self.conv7 = ConvBnDropout(filters=256, strides=1, kernel_size=3, padding="valid", rate=dropout[6])
        self.conv8 = ConvBnDropout(filters=256, strides=1, kernel_size=3, padding="valid", rate=dropout[7])

        self.flat = tf.keras.layers.Flatten()
        self.dense1 = FCDropout(units=512, rate=dropout[8])
        self.dense2 = FCDropout(units=512, rate=dropout[9])

        self.dense3 = FCDropout(units=128, rate=dropout[10])
        self.dense4 = FCDropout(units=128, rate=dropout[11])

        self.dense5 = FCDropout(units=512, rate=dropout[12])

        self.dense = [FCDropout(units=256, rate=dropout[13 + i]) for i in range(10)]

        self.out = [tf.keras.layers.Dense(3) for i in range(4)]
        self.out.append(tf.keras.layers.Dense(1))

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs[0], training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        x = self.conv8(x, training=training)

        x = self.flat(x)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        y = self.dense3(inputs[1], training=training)
        y = self.dense4(y, training=training)

        z = tf.keras.layers.concatenate([x, y], axis=1)
        z = self.dense5(z, training=training)


        branchs = []
        for i in range(4):
            branch = self.dense[i * 2](z, training=training)
            branch = self.dense[i * 2 + 1](branch, training=training)

            branchs.append(self.out[i](branch))

        branch = self.dense[8](x, training=training)
        branch = self.dense[9](branch, training=training)
        branchs.append(self.out[4](branch))

        return branchs, z