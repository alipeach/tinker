import os
import cv2
from flask import Flask, render_template, request, send_file, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

models = {
    'YOLO11n': '/Users/apple/Work/models-yolo11/yolo11n.pt',
    'YOLO11s': '/Users/apple/Work/models-yolo11/yolo11s.pt'
}
current_model = YOLO(models['YOLO11n'])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', models=models.keys())


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': '未提供图片', 'selected_images': [], 'current_index': 0})
    model_name = request.form.get('model')
    global current_model
    current_model = YOLO(models[model_name])
    file = request.files['image']
    img = Image.open(file.stream)
    img = img.convert('RGB')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = current_model(img)
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(annotated_image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return send_file(img_byte_arr, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)