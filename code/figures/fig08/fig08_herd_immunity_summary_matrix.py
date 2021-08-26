import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import itertools
import seaborn as sns

# Summary: assumed vaccine efficacy of 80%

case_16_plus = np.array([[70,              -90],     [90,           -90]])
case_16_labels = np.array([["70% $\geq$ 16s \n 56% of total", "90%\n"], ["90% $\geq$ 16s \n 72% of total", "90%\n"]])

case_12_plus = np.array([[50,              -90], [70,                 -90]])
case_12_labels = np.array([["50% $\geq$ 12s \n 42% of total", "90%\n"], ["70% $\geq$ 12s \n 60% of total", "90%\n"]])

sns.set_context("paper", font_scale=1.8)


def plot_matrix(cm, labels, age_lb, cmap=cm.PRGn_r):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """

  f, ax = plt.subplots(figsize=(6.5,6.0))

  plt.imshow(cm, interpolation='none', cmap=cmap, vmin=-100, vmax=100)
  #plt.colorbar()
  tick_marks = [0, 1]
  plt.xticks(tick_marks, ["min.\nvaccine\ncoverage", "assumed\nvaccine\nefficacy"], rotation=45)
  plt.yticks(tick_marks, ["B.1.1.7","B.1.617.2"])
  plt.plot([0.5, 0.5], [-0.5, 1.5], color="black")
  plt.plot([-0.5, 1.5], [0.5, 0.5], color="black")
  thresh = -100
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, labels[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('dominant variant')
  xlabel_str = 'herd immunity \n requirements ($r_{eff}^{30} <1$)'
  plt.xlabel( xlabel_str)
  ax = plt.gca()
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top') 
  plt.tight_layout()
  plt.show()

# Plot  matrix
plot_matrix(case_16_plus, case_16_labels, "16")
plot_matrix(case_12_plus, case_12_labels, "12")
