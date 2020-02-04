import tensorflow as tf
try:
    import tensorflow.python.keras.layers as layers
except Exception:
    import tensorflow.keras.layers as layers
import numpy as np
import tensorflow_probability as tfp


class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(32)
        self.dense2 = layers.Dense(10)
        self.dense3 = layers.Dense(32)
        self.dense4 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        s = inputs[0]
        a = inputs[1]
        s = layers.LeakyReLU(alpha=0.2)(self.dense1(s))
        a = layers.LeakyReLU(alpha=0.2)(self.dense2(a))
        x = layers.concatenate([s,a])
        x = layers.LeakyReLU(alpha=0.2)(self.dense3(x))
        x = self.dense4(x)

        return x


class Actor(tf.keras.Model):
    def __init__(self, act_dim, discrete=True, act_limit=1):
        super().__init__()
        initer = tf.keras.initializers.Orthogonal()
        # initer = tf.keras.initializers.RandomNormal(stddev=0.2)
        self.act_dim = act_dim
        self.discrete = discrete
        self.act_limit = act_limit
        self.dense1 = layers.Dense(64, activation="relu", kernel_initializer=initer)
        self.dense2 = layers.Dense(64, activation="relu", kernel_initializer=initer)

        if discrete:
            self.dense3 = layers.Dense(act_dim, activation=tf.nn.softmax, kernel_initializer=initer)
        else:
            self.mu = layers.Dense(act_dim, activation="tanh", kernel_initializer=initer)
            # self.sigma = layers.Dense(act_dim, activation=tf.nn.softplus)
            self.sigma = tf.Variable(name="sigma", dtype=tf.float32, trainable=True,
                                     shape=[act_dim, ], initial_value=-0.5*np.ones(act_dim, dtype=np.float32))
            # self.sigma = self.add_weight(name="sigma", dtype=tf.float32, trainable=True,
            #                              shape=[act_dim,], initializer=-0.5*np.ones(act_dim, dtype=np.float32))

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)

        if self.discrete:
            x = self.dense3(x)
            return x

        else:
            mu = self.mu(x)
            sigma = tf.exp(self.sigma)
            sigma = tf.expand_dims(sigma, axis=0)
            sigma = sigma * tf.ones((tf.shape(mu)[0], tf.shape(sigma)[1]))
            # sigma = self.sigma(x)
            return mu*self.act_limit, sigma

    def select_action(self, s):    # discrete: 0-d,  continous: [action_dim]

        if self.discrete:
            x = self.predict(s)
            log_x = tf.math.log(x)

            a = tf.random.categorical(log_x, num_samples=1)
            a = tf.squeeze(a, axis=0)
        else:
            mu, sigma = self.predict(s)
            normal = tfp.distributions.MultivariateNormalDiag(mu, sigma)
            a = normal.sample()
            a = tf.squeeze(a, axis=0)
            a = tf.clip_by_value(a, -self.act_limit, self.act_limit)

            # normal = tfp.distributions.Normal(mu, sigma)
            # a = normal.sample(1)
            # a = tf.squeeze(tf.squeeze(a, axis=0), axis=0)
            # a = tf.clip_by_value(a, -self.act_limit, self.act_limit)

        return a

    def logpi(self, s, a):
        if self.discrete:  # logpi [None, 1]
            logits = self.__call__(s)
            logits = tf.math.log(logits)
            a = tf.squeeze(a, axis=-1)
            a_onehot = tf.one_hot(a, depth=self.act_dim)
            log_pi = tf.reduce_sum(logits*a_onehot, axis=1)
            log_pi = tf.expand_dims(log_pi, axis=-1)
            return log_pi
        else:              # logpi [None, 1]
            mu, sigma = self.__call__(s)
            normal = tfp.distributions.MultivariateNormalDiag(mu, sigma)
            log_pi = normal.log_prob(value=a)
            log_pi = tf.expand_dims(log_pi, axis=-1)
            # normal = tfp.distributions.Normal(mu, sigma)
            # log_pi = normal.log_prob(a)

            return log_pi, mu, sigma


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        initer = tf.keras.initializers.Orthogonal()
        # initer = tf.keras.initializers.RandomNormal(stddev=0.2)
        self.dense1 = layers.Dense(64, activation="relu", kernel_initializer=initer)
        self.dense2 = layers.Dense(64, activation="relu", kernel_initializer=initer)
        self.dense3 = layers.Dense(1, kernel_initializer=initer)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        return x


class PPO:
    def __init__(self, act_dim, epsilon, discrete=True, act_limit=1, lr_a=0.001, lr_c=0.001, c_en=0.01):
        self.discrete = discrete
        self.actor = Actor(act_dim, discrete, act_limit)
        self.old_actor = Actor(act_dim, discrete, act_limit)
        self.critic = Critic()
        self.epsilon = epsilon
        self.opti_a = tf.keras.optimizers.Adam(lr_a)
        self.opti_c = tf.keras.optimizers.Adam(lr_c)
        self.c_en = c_en

    def train_a(self, s, a, adv, lr):

        opti = tf.keras.optimizers.Adam(lr)
        if self.discrete:
            with tf.GradientTape() as tape:
                logpi = self.actor.logpi(s, a)
                old_logpi = self.old_actor.logpi(s, a)

                ratio = tf.math.exp(logpi - old_logpi)
                surr = ratio * adv

                aloss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv)) \
                    - self.c_en*tf.reduce_mean(logpi)

            grad = tape.gradient(aloss, self.actor.trainable_variables)
            opti.apply_gradients(zip(grad, self.actor.trainable_variables))

            return aloss, logpi, old_logpi

        else:
            with tf.GradientTape() as tape:
                logpi, mu, sigma = self.actor.logpi(s, a)
                old_logpi, old_mu, old_sigma = self.old_actor.logpi(s, a)

                ratio = tf.math.exp(logpi-old_logpi)
                surr = ratio*adv

                aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)*adv))\
                        - self.c_en*tf.reduce_mean(logpi)

            grad = tape.gradient(aloss, self.actor.trainable_variables)
            opti.apply_gradients(zip(grad, self.actor.trainable_variables))

            return aloss, logpi, old_logpi, mu, sigma, old_mu, old_sigma

    def update_a(self):
        for var, old_var in zip(self.actor.trainable_variables, self.old_actor.trainable_variables):
            old_var.assign(var)

    def train_v(self, s, vs, lr):
        with tf.GradientTape() as tape:
            v = self.critic(s)
            vloss = tf.reduce_mean(tf.square(v - vs))

        grad = tape.gradient(vloss, self.critic.trainable_variables)
        opti = tf.keras.optimizers.Adam(lr)
        opti.apply_gradients(zip(grad, self.critic.trainable_variables))

        return vloss, v


    def train_v_gae_loss(self, s, oldvs, gae, lr):
        v_targ = oldvs + gae
        with tf.GradientTape() as tape:
            vs = self.critic(s)
            vs_clip = oldvs + tf.clip_by_value(vs - oldvs, -self.epsilon, self.epsilon)

            v_loss_unclip = tf.square(vs - v_targ)
            v_loss_clip = tf.square(vs_clip - v_targ)
            v_loss_mat = tf.maximum(v_loss_clip, v_loss_unclip)

            v_loss = tf.reduce_mean((v_loss_mat))

        grad = tape.gradient(v_loss, self.critic.trainable_variables)
        opti = tf.keras.optimizers.Adam(lr)
        opti.apply_gradients(zip(grad, self.critic.trainable_variables))
        return v_loss, vs
        

    def get_v(self, s):
        return np.squeeze(self.critic(s), axis=0)






