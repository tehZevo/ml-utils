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

def save_plot(x, name, ema_alpha, q=0):
  fig = plt.figure()
  graph_stuff(x, title=name, ema_alpha=ema_alpha)
  lower = np.quantile(x, q)
  upper = np.quantile(x, 1-q)
  plt.ylim(lower, upper)
  plt.savefig(name + ".png")
  plt.close(fig)

def graph_stuff(x, title="", ema_alpha=0.1):
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
