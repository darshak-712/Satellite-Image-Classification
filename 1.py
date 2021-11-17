from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from keras.preprocessing import image
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact_1.0.html")

@app.route("/input",methods=["GET", "POST"])
def input():
    li = ['AnnualCrop', 'Pasture', 'SeaLake', 
        'River', 'Highway', 'Forest', 'Residential', 'Industrial', 'PermanentCrop', 'HerbaceousVegetation']

    if request.method == "POST":
        f = request.files["filename"]
        basepath = os.path.dirname(__file__)
        basepath.replace('//', '/')
        print(basepath)
        file_path = os.path.join(
            basepath, 'static', 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # static_path = os.path.join(
        #     basepath, 'static', secure_filename(f.filename))
        # f.save(static_path)
        li1 = []
        li1.append(f.filename)
        # result_img = file_path.replace('/', '\\')
        # print(result_img)
        result = prediction(file_path)
        print("Predicted class is: ",li[result])
        return render_template("home.html", value = f.filename, val = li1,  label = li[result])

    return render_template("home.html")

@app.route("/output/<filename>")
def output(filename):
    return send_from_directory('static/uploads', filename)

def prediction(img_path):
    img = image.load_img(img_path, target_size=(64,64,3))
    # out_img = preprocessing.normalize(img_tensor) 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    model =tf.keras.models.load_model('model2.h5')
    classes = np.argmax(model.predict(images))
    return classes


if __name__ == "__main__":
    # db.create_all()
    app.run(debug=True)