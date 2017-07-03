# Fractals using tensorflow
# Source: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mandelbrot/index.md
# https://github.com/tobigithub/tensorflow-deep-learning/wiki
import tensorflow as tf
import numpy as np

import pathlib
import PIL
import scipy.ndimage as nd
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--steps', '-s', default=11, type=int)
parser.add_argument('--bounds', '-b', default=3.0, type=float)
parser.add_argument('--startAt', '-st', default=0, type=int)
parser.add_argument('--detailLevel', '-d', default=30, type=int, help='bigger takes longer')
parser.add_argument('--stepSize', '-ss', default=0.004, type=float, help='smaller takes longer and is a bigger output image')
parser.add_argument('--diverged', '-dv', default=4, type=int)
args = parser.parse_args()
print(args)

sess = tf.Session()

def DisplayFractal(a, colorConsts, outputNumber=1,  fmt='jpeg'):
  """Display an array of iteration counts as colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([colorConsts[0]+20*np.cos(a_cyclic),
                        colorConsts[1]+50*np.sin(a_cyclic),
                        colorConsts[2]-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  PIL.Image.fromarray(a).save("%s/output%s.jpeg" % (args.path, outputNumber), fmt)

def generateImage(wiggleFactor, outputNumber, colorConsts):
  tf.global_variables_initializer().run(session=sess)

  # Compute the new values of z: z^2 + x
  zs_ = tf.exp(tf.cos(zs * wiggleFactor))
  # zs**3 - zs**2 + tf.sqrt(zs) + wiggleFactor
  # Have we diverged with this new value?
  not_diverged = tf.abs(zs_) < args.diverged

  # Operation to update the zs and the iteration count.
  #
  # Note: We keep computing zs after they diverge! This is very wasteful!
  #       There are better, if a little less simple, ways to do this.
  #
  step = tf.group(
    zs.assign(zs_),
    ns.assign_add(tf.cast(not_diverged, "float32"))
    )

  for i in range(args.detailLevel): step.run(session=sess)

  DisplayFractal(ns.eval(session=sess), colorConsts=colorConsts, outputNumber=outputNumber)

with tf.device('/gpu:0'):
  # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
  Y, X = np.mgrid[-2:2:args.stepSize, -2:2:args.stepSize]
  Z = X+1j*Y

  xs = tf.constant(Z.astype("complex64"))
  ns = tf.Variable(tf.zeros_like(xs, "float32"))
zs = tf.Variable(xs)

wiggleFactors = np.linspace(-1.0j * args.bounds, 0, num=args.steps)
wiggleLen = len(wiggleFactors)
colorConsts = [10, 80.5, 23.5]

pathlib.Path(args.path).mkdir(exist_ok=True)

for i in range(wiggleLen):
  if i >= args.startAt:
    generateImage(wiggleFactor=wiggleFactors[i], outputNumber=i, colorConsts=colorConsts)
    # generateImage(wiggleFactor=wiggleFactors[i], outputNumber=i, colorConsts=colorConsts)
    if i % int(wiggleLen / 10) == 0:
      print("%s%%" % (i/len(wiggleFactors) * 100))
