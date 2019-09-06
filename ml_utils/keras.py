import tensorflow as tf
#TODO: setup.py extras_require tf

def get_state_variables(model):
  layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.RNN)]
  variables = []
  for layer in layers:
    variables += layer.states
  return variables

def get_states(model):
  #return [K.get_value(s) for s,_ in model.state_updates]
  variables = get_state_variables(model)
  states = [state.numpy() for state in variables]
  return states

def set_states(model, states):
  #for (d,_), s in zip(model.state_updates, states):
  #  K.set_value(d, s)
  variables = get_state_variables(model)
  for variable, state in zip(variables, states):
    variable.assign(state)

def apply_regularization(model, r):
  names = [weight.name for layer in model.layers for weight in layer.weights]
  weights = model.get_weights()

  weights = [weight * (1 - r) if "bias" not in name else weight for name, weight in zip(names, weights)]
  model.set_weights(weights)
