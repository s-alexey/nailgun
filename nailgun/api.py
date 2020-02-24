import os

from flask import Flask, request, jsonify

from skimage.io import imread

from nailgun.model import load, classify

app = Flask(__name__)

model = load(os.environ.get('MODEL_PATH', 'model/'))


@app.route('/predict', methods=['POST'])
def submit():
    # check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'file is missing'}), 400

    file = request.files['image']

    if file:
        image = imread(file)
        prediction = classify(model, image)

        return jsonify(prediction._asdict())

    return jsonify({'error': 'file is empty'}), 400


if __name__ == '__main__':
    app.run(host=os.environ.get('FLASK_RUN_HOST', '0.0.0.0'),
            port=os.environ.get('FLASK_RUN_PORT', '5000'))
