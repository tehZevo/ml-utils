import tensorflow as tf

def dense_stack(input_size, output_size, hidden_sizes=[32, 32], rnn=None,
  acti="tanh", out_acti="tanh", out_lambda=None):
  acti = None if acti.lower() == "none" else acti
  out_acti = None if out_acti.lower() == "none" else out_acti
  inputs = tf.keras.layers.Input(batch_shape=(1, input_size))
  x = inputs
  x = tf.keras.layers.Flatten()(x)
  if rnn is not None:
    x = tf.keras.layers.Reshape((1, input_size))(x)
  for i, size in enumerate(hidden_sizes):
    if rnn is not None:
      last = i == len(hidden_sizes) - 1
      x = rnn(size, activation=acti, return_sequences=not last, stateful=True)(x)
    else:
      x = tf.keras.layers.Dense(size, activation=acti)(x)

  x = tf.keras.layers.Dense(output_size, activation=out_acti)(x)
  if out_lambda is not None:
    x = tf.keras.layers.Lambda(out_lambda)(x)

  model = tf.keras.Model(inputs, x)

  return model
