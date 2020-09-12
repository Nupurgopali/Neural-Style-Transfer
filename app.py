import numpy as np
from flask import Flask, request, jsonify, render_template,Markup,flash
import pickle
import pandas as pd
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
from PIL import Image
import glob


app = Flask(__name__,template_folder='template')

#app.config["SECRET_KEY"] = 'secremessage'
#APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/upload_cimage',methods=['GET','POST'])
def upload_cimage():
    img = request.files['file']
    iname=img.filename
    file_ext = os.path.splitext(iname)[1]
    

    paths=r'C:/Users/Lenovo/Desktop/nst/content_images'
    img.save(os.path.join(paths,iname)) 
    
    

    content_image_url=os.path.join(paths,iname)
    style_image_url=r'C:/Users/Lenovo/Desktop/nst/image/filter3.jpg'
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
    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_img_size)
    new_model = tf.keras.models.load_model('model/my_model')
    outputs = new_model(content_image, style_image)
    stylized_image = outputs[0]
    pil_img = tf.keras.preprocessing.image.array_to_img(stylized_image[0])
    pil_img.save(r'C:/Users/Lenovo/Desktop/nst/static/gen_image/{}'.format(iname)) 
    name=r'gen_image/{}'.format(iname)
    

    
    
    return render_template("complete.html",user_image=name)
@app.route('/delete_img',methods=['GET','POST'])
def delete_img():
    test = 'C:/Users/Lenovo/Desktop/nst/content_images/*'
    test2='C:/Users/Lenovo/Desktop/nst/static/gen_image/*'
    r = glob.glob(test)
    k=glob.glob(test2)
    for i in r:
        os.remove(i)
        
    for j in k:
        os.remove(j)
    return render_template("index.html")



    
if __name__ == "__main__":
    app.run(debug=False)  
