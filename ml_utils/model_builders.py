import tensorflow as tf

def dense_stack(input_size, output_size, hidden_sizes=[32, 32], rnn=None,
  acti="tanh", out_acti="tanh", out_lambda=None):
  acti = None if acti.lower() == "none" else acti
  out_acti = None if out_acti.lower() == "none" else out_acti
  #TODO: dont force batch shape here
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

#TODO
def dense_autoencoder(input_size, latent_size, hidden_sizes=[32, 16], acti="tanh", input_range=None):
  #latent space will be constrained to -1..1 range
  encoder = dense_stack(input_size, latent_size, hidden_sizes, acti=acti,
    out_acti="tanh")

  out_acti = "linear" if input_range is None else "sigmoid"
  out_lambda = (lambda x: x * (input_range[1] - input_range[0]) + input_range[0]) if input_range is not None else None
  decoder = dense_stack(latent_size, input_size, reversed(hidden_sizes),
    acti=acti, out_acti=out_acti, out_lambda=out_lambda)

  inputs = encoder.input
  x = decoder(encoder(inputs))
  model = tf.keras.Model(inputs, x)
  return model
