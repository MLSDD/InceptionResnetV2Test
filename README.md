Inception V2 minimal example
============================

Minimal InceptionResnetV2 implementation based on https://github.com/tensorflow/models/tree/master/slim pretrained model. Its main purpose is to test out the implementation before attempting to do transfer learning.

Input format
------------

Network input is BGR and in range of 0-255 to match OpenCV's image representation.