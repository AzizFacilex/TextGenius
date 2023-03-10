import keras_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify


app = Flask(__name__)
pipeline = keras_ocr.pipeline.Pipeline()


def sort_words_by_position(words):
    # Sort the list of words by the x-coordinate of the top-left corner of each bounding box
    sorted_words = sorted(words, key=lambda w: w['box'][0][0])

    # Initialize an empty list to store the sorted words
    result = []

    # Loop through the sorted list of words
    for word in sorted_words:
        # If the list of sorted words is empty, append the current word to it
        if not result:
            result.append(word)
        else:
            last_word = result[-1]
            # Compare the x-coordinate of the top-left corner of the current word's bounding box
            # with that of the last word in the sorted list
            if word['box'][0][0] > last_word['box'][0][0]:
                result.append(word)
            else:
                # Loop through the sorted list backwards and find the index at which the current word
                # should be inserted based on its x-coordinate
                for i in range(len(result)-1, -1, -1):
                    if word['box'][0][0] > result[i]['box'][0][0]:
                        result.insert(i+1, word)
                        break
                else:
                    result.insert(0, word)

    # Return the sorted list of words
    return result


@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['image']

    image_data = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # gray = cv2.fastNlMeansDenoising(gray, h=10)
    # binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=100, maxLineGap=10)

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
    images = [cv2.warpAffine(image, rotation_matrix, (cols, rows))]
    prediction_groups = pipeline.recognize(images)

    predicted_image_1 = prediction_groups[0]
    predicted_image_1_sorted = sorted(predicted_image_1, key=lambda x: (
        x[1][0][1], x[1][0][0]))  # write output text to file in correct order
    phrases = []

    current_phrase = []

    for i in range(len(predicted_image_1_sorted)):
        word = predicted_image_1_sorted[i][0]
        box = predicted_image_1_sorted[i][1]

        if i == 0:
            current_phrase.append({'word': word, 'box': box})
        else:
            prev_box = predicted_image_1_sorted[i-1][1]

            # assume words within 5 pixels of each other are on the same line
            if abs(box[0][1] - prev_box[0][1]) <= 5:
                current_phrase.append({'word': word, 'box': box})
            else:
                # Sort the words in the current phrase by their x-coordinates
                sorted_words = sort_words_by_position(current_phrase)
                phrases.append(' '.join(w['word'] for w in sorted_words))
                current_phrase = [{'word': word, 'box': box}]

        if i == len(predicted_image_1_sorted) - 1:
            # Sort the words in the current phrase by their x-coordinates
            sorted_words = sort_words_by_position(current_phrase)
            phrases.append(' '.join(w['word'] for w in sorted_words))

    response = {'text': phrases}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
