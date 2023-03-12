import cv2
import numpy as np
from flask import Flask, request, jsonify
import string

from processing import *

app = Flask(__name__)


def cleanList(max_list):
    max_list['text'] = [
        word for word in max_list['text'] if word != '']
    max_list['text'] = [word.strip()
                        for word in max_list['text'] if word.strip() != '']
    max_list['text'] = [
        word for word in max_list['text'] if not all(char in string.punctuation for char in word)]


def readImage(file):
    image_data = np.fromstring(file.read(), np.uint8)
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)


@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['image']
    # summaryText = request.form['text']

    image = readImage(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = preprocessAdaptiveThreshold(image, gray)
    recognisedPhrases = [extract(image)]

    image = preprocessThreshold(image, gray)
    recognisedPhrases.append(extract(image))

    image = preprocessDenoising(image, gray)
    recognisedPhrases.append(extract(image))

    max_list = max(recognisedPhrases, key=lambda x: sum(x['conf']))
    cleanList(max_list)

    jobText = ' '.join(max_list['text'])

    # TODO
    # gptResponse = sendToGpt(summaryText, jobText)

    response = {'text': jobText}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
