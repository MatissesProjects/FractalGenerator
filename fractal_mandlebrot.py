# Mandelbrot using tensorflow
# Source: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mandelbrot/index.md
# https://github.com/tobigithub/tensorflow-deep-learning/wiki
import tensorflow as tf
import numpy as np

import pathlib
import PIL
import scipy.ndimage as nd

path = "./cubic3"
steps = 1000

sess = tf.Session()

startValue = 789
detailLevel = 10
  

def DisplayFractal(a, colorConsts, outputNumber=1,  fmt='jpeg'):
  """Display an array of iteration counts as a
     colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([colorConsts[0]+20*np.cos(a_cyclic),
                        colorConsts[1]+50*np.sin(a_cyclic),
                        colorConsts[2]-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  PIL.Image.fromarray(a).save("%s/output%s.jpeg" % (path, outputNumber), fmt)

def generateImage(wiggleFactor, outputNumber, colorConsts):
  tf.global_variables_initializer().run(session=sess)
  # sq = tf.sqrt(xs) + wiggleFactor
  # Compute the new values of z: z^2 + x
  zs_ = tf.exp(tf.cos(zs * wiggleFactor))
  # zs**3 - zs**2 + tf.sqrt(zs) + wiggleFactor
  # Have we diverged with this new value?
  not_diverged = tf.abs(zs_) < 4

  # Operation to update the zs and the iteration count.
  #
  # Note: We keep computing zs after they diverge! This
  #       is very wasteful! There are better, if a little
  #       less simple, ways to do this.
  #
  step = tf.group(
    zs.assign(zs_),
    ns.assign_add(tf.cast(not_diverged, "float32"))
    )

  for i in range(detailLevel): step.run(session=sess)

  DisplayFractal(ns.eval(session=sess), colorConsts=colorConsts, outputNumber=outputNumber)

with tf.device('/gpu:0'):
  # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
  Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]
  Z = X+1j*Y

  xs = tf.constant(Z.astype("complex64"))
  ns = tf.Variable(tf.zeros_like(xs, "float32"))
  
zs = tf.Variable(xs)

wiggleFactors = np.linspace(-2.5j, 2.5j, num=steps)
wiggleLen = len(wiggleFactors)
colorConsts = [10, 80.5, 23.5]

pathlib.Path(path).mkdir(exist_ok=True)

for i in range(wiggleLen):
  # TODO: add a continue flag
  if i > startValue:
    generateImage(wiggleFactor=wiggleFactors[i], outputNumber=i, colorConsts=colorConsts)
    # generateImage(wiggleFactor=wiggleFactors[i], outputNumber=i, colorConsts=colorConsts)
    if i % int(wiggleLen / 10) == 0:
      print("%s%%" % (i/len(wiggleFactors) * 100))
### END