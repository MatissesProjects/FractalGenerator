# Mandelbrot using tensorflow
# Source: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mandelbrot/index.md
# https://github.com/tobigithub/tensorflow-deep-learning/wiki
import tensorflow as tf
import numpy as np

import pathlib
import PIL.Image
from io import BytesIO
import argparse
import scipy.misc as saver

from IPython.display import Image

path = "./ecubic3"
steps = 400

sess = tf.Session()

startValue = 0
detailLevel = 100

def DisplayFractal(a, colorConsts, saveIndex, outputNumber=1,  fmt='jpeg'):
  """Display an array of iteration counts as a
     colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([colorConsts[0]+20*np.cos(a_cyclic),
                        colorConsts[1]+50*np.sin(a_cyclic),
                        colorConsts[2]-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = [150, 200, 155]
  a = img
  a = abs(a)
  a %= 255
  a = np.uint8(np.clip(a, 0, 255))
  # PIL.Image.fromarray(a).save(, fmt)
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  saver.imsave("%s/output%s_%s.jpeg" % (path, outputNumber, saveIndex), a)
  # display(Image(data=f.getvalue()))

def generateImage(wiggleFactor, xyAxisMove, outputNumber, colorConsts, saveIndex):
  tf.global_variables_initializer().run(session=sess)
  # sq = tf.sqrt(xs) + wiggleFactor
  # Compute the new values of z: z^2 + x
  zs_ = tf.exp(tf.cos(zs * wiggleFactor)) + zs**xyAxisMove * wiggleFactor
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

  DisplayFractal(ns.eval(session=sess), colorConsts=colorConsts, outputNumber=outputNumber, saveIndex=saveIndex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animation generator')
    parser.add_argument('-axisModifier', type=float, default=0, help='float for x axis movement')
    parser.add_argument('-saveIndex', type=int, default=0, help='int for save index, into x axis')
    args = parser.parse_args()
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    Y, X = np.mgrid[-1:1:0.01, -1:1:0.01]
    Z = X+1j*Y

    xs = tf.constant(Z.astype("complex64"))
    ns = tf.Variable(tf.zeros_like(xs, "float32"))
    zs = tf.Variable(xs)

    wiggleFactors = np.linspace(-2j, 2j, num=steps)
    wiggleLen = len(wiggleFactors)
    colorConsts = [10, 80.5, 23.5]

    pathlib.Path(path).mkdir(exist_ok=True)

    print(args.axisModifier)
    for i in range(wiggleLen):
      if i > startValue:
        generateImage(wiggleFactor=wiggleFactors[i], xyAxisMove=args.axisModifier, outputNumber=i,
                      colorConsts=colorConsts, saveIndex=args.saveIndex)
        # generateImage(wiggleFactor=wiggleFactors[i], outputNumber=i, colorConsts=colorConsts)
        if i % int(wiggleLen / 10) == 0:
          print("%s%%" % (i/len(wiggleFactors) * 100))
    sess.close()
    ### END
