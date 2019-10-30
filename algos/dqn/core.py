import tensorflow as tf
from tensorflow.python.keras import layers, optimizers
# import tensorflow.contrib.layers as layers


def nature_dqn(num_actions, input_shape=((84, 84, 4))):

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, [8, 8], strides=4, input_shape=input_shape, activation="relu", data_format='channels_last'))
    model.add(layers.Conv2D(64, [4, 4], strides=2, activation="relu"))
    model.add(layers.Conv2D(64, [3, 3], strides=1, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(num_actions))

    model.compile(optimizer=optimizers.Adam(0.0001),
                  loss="mse",
                  metrics=["mae"])

    model_target = tf.keras.Sequential()
    model_target.add(layers.Conv2D(32, [8, 8], strides=4, input_shape=input_shape, activation="relu", data_format='channels_last'))
    model_target.add(layers.Conv2D(64, [4, 4], strides=2, activation="relu"))
    model_target.add(layers.Conv2D(64, [3, 3], strides=1, activation="relu"))
    model_target.add(layers.Flatten())
    model_target.add(layers.Dense(512, activation="relu"))
    model_target.add(layers.Dense(num_actions))

    return model, model_target


def mlp_dqn(num_actions, input_shape, hidden_sizes=(400,300), activation="relu"):

    model = tf.keras.Sequential()
    model.add(layers.Dense(hidden_sizes[0], input_shape=input_shape, activation=activation))
    for i in hidden_sizes[1:-1]:
        model.add(layers.Dense(hidden_sizes[i], activation=activation))

    model.add(layers.Dense(num_actions))

    model.compile(optimizer=optimizers.Adam(),
                  loss="mse",
                  metrics=["mae"])

    model_target = tf.keras.Sequential()
    model_target.add(layers.Dense(hidden_sizes[0], input_shape=input_shape, activation=activation))
    for i in hidden_sizes[1:-1]:
        model_target.add(layers.Dense(hidden_sizes[i], activation=activation))

    model_target.add(layers.Dense(num_actions))

    return model, model_target

# def nature_dqn_contrib(input, actions, scope):
#
#     with tf.variable_scope(scope):
#         x = input
#         x = layers.convolution2d(x, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
#         x = layers.convolution2d(x, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
#         x = layers.convolution2d(x, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
#
#         x = layers.flatten(x)
#         x = layers.fully_connected(x, num_outputs=512, activation_fn=tf.nn.relu)
#         x = layers.fully_connected(x, num_outputs=actions, activation_fn=None)
#
#         return x
