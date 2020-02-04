import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class ConvBnDropout(tf.keras.Model):
    def __init__(self, filters, strides, kernel_size, padding="valid", rate=0.2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(1-rate)
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
        self.drop = tf.keras.layers.Dropout(1-rate)
        self.relu = tf.keras.layers.ReLU()

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

    def call(self, inputs, training=False):
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


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = [tf.keras.layers.Dense(256, activation="relu") for i in range(8)]
        self.dense2 = [tf.keras.layers.Dense(1) for i in range(4)]

    def call(self, inputs, training=False, mask=None):
        x = inputs

        branches = []
        for i in range(4):
            branch = self.dense1[i*2](x)
            branch = self.dense1[i*2+1](branch)

            branch = self.dense2[i](branch)
            branches.append(branch)

        x = tf.stack(branches, axis=1)             # [None, 4, 1]
        return x


class Actor(tf.keras.Model):
    def __init__(self, act_dim, dropout=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5):
        super().__init__()
        self.act_dim = act_dim

        self.emd = ActorEmd(dropout)
        self.dense = [FCDropout(units=256, rate=dropout[13+i]) for i in range(10)]

        self.mu = [[tf.keras.layers.Dense(1, activation="tanh"), tf.keras.layers.Dense(2, activation="sigmoid")]
                   for i in range(4)]
        self.sigma = tf.Variable(name="sigma", dtype=tf.float32, trainable=True,
                                     shape=[act_dim, ], initial_value=-0.5*np.ones(act_dim, dtype=np.float32))

    def call(self, inputs, training=False, mask=None):
        x, z = self.emd(inputs, training=training)

        branchs = []
        for i in range(4):
            branch = self.dense[i*2](z, training=training)
            branch = self.dense[i*2+1](branch, training=training)

            branch1 = self.mu[i][0](branch, training=training)
            branch2 = self.mu[i][1](branch, training=training)
            branch = tf.keras.layers.concatenate([branch1, branch2], axis=1)

            branchs.append(branch)

        mu = tf.stack(branchs, axis=1)                # [None, 4, 3]
        sigma = tf.exp(self.sigma)
        sigma = tf.expand_dims(sigma, axis=0)
        sigma = sigma * tf.ones((tf.shape(mu)[0], tf.shape(sigma)[1]))

        return mu, sigma

    def select_action(self, s, condition):
        mu, sigma = self.predict(s)                   # [None, 4, 3]   [None, 3]
        condition = tf.stack([tf.range(1), [condition]], axis=1)     # [None, 2]

        mu_c = tf.gather_nd(mu, condition)            # [None, 3]
        normal = tfp.distributions.MultivariateNormalDiag(mu_c, sigma)
        a = normal.sample()

        a = tf.squeeze(a, axis=0).numpy()
        a[0] = np.clip(a[0], -1, 1)
        a[1] = np.clip(a[1], 0, 1)
        a[2] = np.clip(a[2], 0, 1)

        return a

    def logpi(self, s, a, c):
        mu, sigma = self.__call__(s, training=True)                # [None, 4, 3]   [None, 3]
        b = tf.shape(s[0])[0]
        c = tf.stack([tf.range(b), c], axis=1)        # [None, 2]

        mu_c = tf.gather_nd(mu, c)            # [None, 3]
        normal = tfp.distributions.MultivariateNormalDiag(mu_c, sigma)
        log_pi = normal.log_prob(value=a)
        log_pi = tf.expand_dims(log_pi, axis=-1)               #[None, 1]
        return log_pi, mu_c, sigma


class PPO(tf.keras.Model):
    def __init__(self, act_dim, epsilon, lr_a=0.001, lr_c=0.001, c_en=0.01):
        super().__init__()
        self.actor = Actor(act_dim)
        self.old_actor = Actor(act_dim)
        self.critic = Critic()
        self.epsilon = epsilon

        self.opti_a = tf.keras.optimizers.Adam(lr_a)
        self.opti_c = tf.keras.optimizers.Adam(lr_c)
        self.c_en = c_en

    def train_a(self, s, a, adv):
        with tf.GradientTape() as tape:
            logpi, mu, sigma = self.actor.logpi([s["img"], s["speed"]], a, s["direction"])
            old_logpi, old_mu, old_sigma = self.old_actor.logpi([s["img"], s["speed"]], a, s["direction"])

            ratio = tf.math.exp(logpi-old_logpi)
            surr = ratio*adv

            aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)*adv))\
                    - self.c_en*tf.reduce_mean(logpi)

        grad = tape.gradient(aloss, self.actor.trainable_variables)
        self.opti_a.apply_gradients(zip(grad, self.actor.trainable_variables))

        return aloss, logpi, old_logpi, mu, sigma, old_mu, old_sigma

    def train_v(self, s, vs):
        img = s["img"]
        speed = s["speed"]
        c = s["direction"]
        c = tf.stack([tf.range(tf.shape(img)[0]), c], axis=1)

        with tf.GradientTape() as tape:
            x, z = self.actor.emd([img, speed], training=True)
            v = self.critic(z)
            v = tf.gather_nd(v, c)          # [None, 1]

            v_loss = tf.keras.losses.mse(vs, v)
        var = self.actor.emd.trainable_variables + self.critic.trainable_variables
        gra = tape.gradient(v_loss, var)
        self.opti_c.apply_gradients(zip(gra, var))
        return v_loss, v

    def getV(self, s):             # only use in None=1
        img = s["img"]
        speed = s["speed"]
        c = s["direction"]
        # c = tf.stack([tf.range(tf.shape(img)[0], c)])

        x, z = self.actor.emd([img, speed], training=True)
        v = self.critic(z)              # [None, 4, 1]
        # v = tf.gather_nd(v, c)          # [None, 1]
        v = v[:, c]
        v = tf.squeeze(v, axis=0)          # [1,]

        return v

    def update_a(self):
        for var, old_var in zip(self.actor.trainable_variables, self.old_actor.trainable_variables):
            old_var.assign(var)







