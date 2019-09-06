import matplotlib.pyplot as plt
import math
import numpy as np

def ema(x, alpha=0.01):
  mean = []
  mad = []
  for val in x:
    val = 0 if math.isnan(val) or val is None else val #ree

    if len(mean) == 0:
      mean.append(val)
      mad.append(0)
    else:
      diff = val - mean[-1]
      mean.append(mean[-1] + diff * alpha)
      mad.append(mad[-1] + (abs(diff) - mad[-1]) * alpha)
  mean = np.array(mean)
  mad = np.array(mad)
  return (mean, mad)

def graph_stuff(x, title="", ema_alpha=0.1):
  #TODO: update to support multiple lines
  plt.title(title)
  col = "C{}".format(0 % 10)
  plt.plot(x, color=col, alpha = 0.25)
  mean, mad = ema(x, ema_alpha)

  plt.fill_between(range(len(mean)), mean + mad, mean - mad, color=col, alpha=0.25)
  plt.plot(mean, color=col, linestyle="dashed")
  #plt.show()

def viz_weights(weights, filename):
  fig = plt.figure()
  weights = np.concatenate([x.flatten() for x in weights])
  sq = int(np.ceil(np.sqrt(len(weights))))
  sqsq = sq * sq
  weights = np.concatenate([weights, np.zeros(sqsq-len(weights))])
  weights = np.reshape(weights, (sq, sq))

  sns.heatmap(weights)
  plt.savefig(filename)
  plt.close(fig)
