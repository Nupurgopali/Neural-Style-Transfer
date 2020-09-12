import pickle
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt

content_image=''
style_image=''
def load_image(image, image_size=(256, 256), preserve_aspect_ratio=True):
        img = plt.imread(image).astype(np.float64)[np.newaxis, ...]
        if img.max() > 1.0:
            img = img / 255.
        if len(img.shape) == 3:
            img = tf.stack([img, img, img], axis=-1)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img
output_image_size = 384 
content_img_size = (output_image_size, output_image_size)
style_img_size = (256, 256)

new_model = tf.keras.models.load_model('model/my_model')
outputs = new_model(content_image, style_image)

stylized_image = outputs[0]
def main(content_image_path, style_image_path):
    content_image_path  = content_image_path
    style_image_path =  style_image_path
    content_image = load_image(content_image_path, content_img_size)
    style_image = load_image(style_image_path, style_img_size)
    #style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')