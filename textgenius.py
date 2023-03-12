import cv2
import numpy as np
from flask import Flask, request, jsonify
import pytesseract
import string

app = Flask(__name__)


def extractAndSort(image):
    tesseract_results = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT)

    return tesseract_results


def rotate(image, lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -45 <= angle <= 45:
            angles.append(angle)

    average_angle = np.mean(angles)

    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (cols/2, rows/2), average_angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))


@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['image']

    image_data = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=100, maxLineGap=10)
    image = rotate(image, lines)
    recognisedPhrases = [extractAndSort(image)]

    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=100, maxLineGap=10)
    image = rotate(image, lines)
    recognisedPhrases.append(extractAndSort(image))

    gray = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=100, maxLineGap=10)
    image = rotate(image, lines)
    recognisedPhrases.append(extractAndSort(image))

    max_list = max(recognisedPhrases, key=lambda x: sum(x['conf']))

    max_list['text'] = [
        word for word in max_list['text'] if word != '']
    max_list['text'] = [word.strip()
                        for word in max_list['text'] if word.strip() != '']
    max_list['text'] = [
        word for word in max_list['text'] if not all(char in string.punctuation for char in word)]
    response = {'text': ' '.join(max_list['text'])}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
