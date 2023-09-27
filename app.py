from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io
import os
import shutil
import cv2
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import r2_score
from tensorflow.keras.preprocessing import image
from tensorflow.image import ResizeMethod
import cv2

app = Flask(__name__)
session = {}


def isValidFile(file_name):
    accepted_files = ['jpg', 'jpeg', 'png']
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in accepted_files


def detectCyclone(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    model = torch.hub.load('./yolov5', 'custom', path='models/best.pt', force_reload=True, source='local')

    results = model(img, size=640)

    image = results.crop(save=False)[0]['im']
    cv2.imwrite('./static/roi_images/image.jpg', image)

    results = list(results.xyxy[0][:, :-1].numpy()[0])
    
    confidence = results[-1]

    results = [int(x) for x in results[:4]]
    results.append(confidence)

    return results

def intensityModel():
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in vgg_model.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    
    return model

def estimateIntensity():
    model = session['intensity_model']
    model.load_weights('./models/' + 'vgg16.h5')
    img = image.load_img('./static/roi_images/image.jpg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img/255)[0][0]*100

    return pred

@app.route('/')
def getHome():
    return render_template('Home.html')

@app.route('/getCycloneDetectionPage')
def getCycloneDetectionPage():
    return render_template('Cyclone_Detection.html', error=-1)

@app.route('/upload_image', methods=['POST'])
def uploadImage():
    file = request.files['cyclone_image']
    file_name = secure_filename(file.filename)

    if isValidFile(file_name):

        img_bytes = file.read()
        result = detectCyclone(img_bytes)
        intensity = estimateIntensity()

        image_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        cv2.imwrite('./static/original_images/image.jpg', image)

        start_point = (result[0], result[1])
        end_point = (result[2], result[3])

        image = cv2.rectangle(image, end_point, start_point, (0, 0, 255), 2)

        (w, h), _ = cv2.getTextSize(f'Confidence: {format(result[-1], ".2f")} Intensity: {format(intensity, ".2f")}', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        image = cv2.rectangle(image, (result[0], result[1] - 20), (result[0] + w, result[1]), (0, 0, 255), -1)

        image = cv2.putText(image, f'Confidence: {format(result[-1], ".2f")} Intensity: {format(intensity, ".2f")}', (result[0], result[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        cv2.imwrite('./static/cyclone_detected_images/image.jpg', image)

        file_name = 'image.jpg'
        category = ''
        status = 'Detected'

        if 0 < intensity < 74:
            category = "Tropical Depression"
        elif 74 <= intensity and intensity < 95:
            category = "Tropical Storm"
        elif 95 <= intensity and intensity < 111:
            category = "Category 1 Hurricane"
        elif 111 <= intensity and intensity < 130:
            category = "Category 2 Hurricane"
        elif 130 <= intensity and intensity < 157:
            category = "Category 3 Hurricane (Major)"
        elif 157 <= intensity < 178:
            category = "Category 4 Hurricane (Major)"
        else:
            category = "Category 5 Hurricane (Major)"

        return render_template('Cyclone_Detection.html', file_name=file_name, status = status, category = category, intensity = intensity, error=0)

    return render_template('Cyclone_Detection.html', error=1)


if __name__ == '__main__':
    session['intensity_model'] = intensityModel()
    app.run(debug=True)