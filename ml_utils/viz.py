import math
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def ema(x, alpha=0.01):
  mean = []
  variance = []
  for val in x:
    val = 0 if math.isnan(val) or val is None else val #ree

    if len(mean) == 0:
      mean.append(val)
      variance.append(0)
    else:
      diff = val - mean[-1]
      mean.append(mean[-1] + diff * alpha)
      variance.append(variance[-1] + (abs(diff) - variance[-1]) * alpha)
  mean = np.array(mean)
  variance = np.array(variance)
  return (mean, variance)

def rolling_window(a, window):
  shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
  strides = a.strides + (a.strides[-1],)
  return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def graph_stuff(x, title="", smoothness=0.1, draw_raw=True):
  x = np.array(x)
  if x.ndim == 1:
    x = np.expand_dims(x, 0)
  else:
    x = x.transpose() #sanity

  plt.title(title)

  count = int(np.ceil(x.shape[1] * smoothness))
  offset_xs = [i + count - 1 for i in range(x.shape[1] - count + 1)]
  for i in range(x.shape[0]):
    data = x[i]
    col = "C{}".format(i % 10)
    if draw_raw:
      plt.plot(data, color=col, alpha = 0.25, label=None)
    mean = np.mean(rolling_window(data, count), -1)
    std = np.std(rolling_window(data, count), -1, ddof=0) #=/

    plt.fill_between(offset_xs, mean + std, mean - std, color=col, alpha=0.25)
    plt.plot(offset_xs, mean, color=col, label=i)#, linestyle="dashed")

  if x.shape[0] > 1:
    plt.legend()

def save_plot(x, name, smoothness=0.1, q=0):
  fig = plt.figure()
  #graph_stuff(x, title=name, smoothness=smoothness)
  graph_stuff_ema(x, title=name, ema_alpha=smoothness)
  lower = np.quantile(x, q)
  upper = np.quantile(x, 1-q)
  plt.ylim(lower, upper)
  plt.savefig(name + ".png")
  plt.close(fig)

#old EMA based version
def graph_stuff_ema(x, title="", ema_alpha=0.1):
  x = np.array(x)
  if x.ndim == 1:
    x = np.expand_dims(x, 0)
  else:
    x = x.transpose() #sanity

  plt.title(title)

  for i in range(x.shape[0]):
    data = x[i]
    col = "C{}".format(i % 10)
    plt.plot(data, color=col, alpha = 0.25, label=None)
    mean, variance = ema(data, ema_alpha)

    plt.fill_between(range(len(mean)), mean + variance, mean - variance, color=col, alpha=0.25)
    plt.plot(mean, color=col, label=i)#, linestyle="dashed")

  if x.shape[0] > 1:
    plt.legend()

def viz_weights(weights, filename):
  fig = plt.figure()
  weights = np.concatenate([x.flatten() for x in weights])
  sq = int(np.ceil(np.sqrt(len(weights))))
  sqsq = sq * sq
  weights = np.concatenate([weights, np.zeros(sqsq-len(weights))])
  weights = np.reshape(weights, (sq, sq))

  sns.heatmap(weights)#plt.imshow(diff)
  plt.savefig(filename)
  plt.close(fig)
