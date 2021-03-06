import flask
import os
import pandas as pd
import skimage
import skimage.io
import skimage.transform
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import copy
app = flask.Flask(__name__, template_folder='templates')

model = keras.models.load_model("tumor_model2.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def callModel(img):

    X = image.img_to_array(img)
    X = np.expand_dims(X,axis = 0)
    images = np.vstack([X])
    val = model.predict(images)

    return val

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename.replace(" ", "")))
            # Read the image using skimage

            img = skimage.io.imread("static/uploads/"+str(filename).replace(" ", ""))

            # Resize the image to match the input the model will accept
            img = skimage.transform.resize(img, (150, 150, 1))

            # Get prediction of image from classifier
            predictions = callModel(img)

            # Get the value of the prediction
            prediction = predictions[0]


            return flask.render_template('index.html', prediction =  "<b>Prediction for input image is</b> <br><b>Giloma:</b> "+str(round(prediction[0]*100, 5))+"%" +
                                                                     "<br><b>Meningioma:</b> "+str(round(prediction[1]*100, 5))+"%" +
                                                                     "<br><b>Pituitary:</b> "+str(round(prediction[3]*100, 5))+"%" +
                                                                     "<br><b>No Tumor:</b> "+str(round(prediction[2]*100, 5))+"%", image="static/uploads/"+str(filename).replace(" ", ""))

    return(flask.render_template('index.html'))


if __name__ == '__main__':
    app.run(debug=True)
