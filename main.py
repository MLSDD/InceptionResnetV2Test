#!/usr/bin/python

import tensorflow as tf
from InceptionResnetV2 import *
import numpy as np
import cv2
import sys
import ImageNet
import Model

Model.download()

if len(sys.argv)<2:
    print("Usage: ")
    print("    "+sys.argv[0]+" <image>")
    sys.exit(-1)


w=299
h=299

inPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 3])
net = InceptionResnetV2("googleNet",inPlaceholder,isTraining=False)

out = net.getOutput("Predictions")

with tf.Session() as sess:
    print("Importing googleNet")
    net.importWeights(sess, Model.FILENAME)
    print("Done.")

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Failed to load image '"+sys.argv[1]+"'")
        sys.exit(-1)
    img = cv2.resize(img, (w,h))

    res = sess.run(out, feed_dict={inPlaceholder: np.expand_dims(img, axis=0)})[0]

    tops = res.argsort()[-5:].tolist()[::-1]
    for t in tops:
        label = ImageNet.labels[t-1] if t>0 else "background"
        print("    "+ImageNet.labels[t-1]+(": %.2f %%" % (res[t]*100.0)))
    