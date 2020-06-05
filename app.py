# -*- coding: UTF-8 -*-
import os
import sys
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, make_response, request

from mrcnn.config import Config
import mrcnn.model as modellib

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)


class InferenceConfig(Config):
    NAME = "Sign"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 50
    BACKBONE = "resnet50"


@app.route('/', methods=['GET'])
def index():
    resp = make_response(render_template('index.html'))

    return resp


@app.route('/results', methods=['GET'])
def result():
    results = filter(lambda x: ".png" in x, os.listdir(ROOT_DIR + "/static/results"))
    resp = make_response(render_template('results.html', imgs=results))

    return resp


@app.route('/api/getResult', methods=['POST'])
def get_result():
    resp = dict()
    resp["ok"] = True

    image = Image.open(request.files['image'])

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=ROOT_DIR)

    model.load_weights("model.h5", by_name=True)

    r = model.detect([np.array(image)], verbose=1)[0]

    file_name = str(uuid.uuid4())
    for i in range(len(r['rois'])):
        mask_result = Image.fromarray(r['masks'][:, :, i])

        mask_result.save(ROOT_DIR + "/static/results/" + file_name + "_" + str(i) + ".png")

    image.save(ROOT_DIR + "/static/results/" + file_name + ".png")

    return make_response(resp)


if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
