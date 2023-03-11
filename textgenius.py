import keras_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import Levenshtein
import difflib
import spacy

nlp = spacy.load('en_core_web_sm')  # load the English language model
app = Flask(__name__)
pipeline = keras_ocr.pipeline.Pipeline()


def sort_words_by_position(words):
    sorted_words = sorted(words, key=lambda w: w['box'][0][0])

    result = []

    for word in sorted_words:
        if not result:
            result.append(word)
        else:
            last_word = result[-1]
            if word['box'][0][0] > last_word['box'][0][0]:
                result.append(word)
            else:
                for i in range(len(result)-1, -1, -1):
                    if word['box'][0][0] > result[i]['box'][0][0]:
                        result.insert(i+1, word)
                        break
                else:
                    result.insert(0, word)

    return result


def extractAndSort(image):
    prediction_groups = pipeline.recognize([image])

    predicted_image_1 = prediction_groups[0]
    predicted_image_1_sorted = sorted(predicted_image_1, key=lambda x: (
        x[1][0][1], x[1][0][0]))
    phrases = []

    current_phrase = []

    for i in range(len(predicted_image_1_sorted)):
        word = predicted_image_1_sorted[i][0]
        box = predicted_image_1_sorted[i][1]

        if i == 0:
            current_phrase.append({'word': word, 'box': box})
        else:
            prev_box = predicted_image_1_sorted[i-1][1]

            if abs(box[0][1] - prev_box[0][1]) <= 5:
                current_phrase.append({'word': word, 'box': box})
            else:
                sorted_words = sort_words_by_position(current_phrase)
                phrases.append(' '.join(w['word'] for w in sorted_words))
                current_phrase = [{'word': word, 'box': box}]

        if i == len(predicted_image_1_sorted) - 1:
            sorted_words = sort_words_by_position(current_phrase)
            phrases.append(' '.join(w['word'] for w in sorted_words))

    return phrases


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


def get_best_words(phrases):
    best_words = []
    for i in range(len(phrases[0])):
        possible_words = [phrase[i] for phrase in phrases]
        best_word = max(possible_words, key=lambda x: difflib.SequenceMatcher(
            None, x, possible_words[0]).ratio())
        best_words.append(best_word)
    return best_words


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

    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
    #                         threshold=50, minLineLength=100, maxLineGap=10)
    # image = rotate(image, lines)
    # recognisedPhrases.append(extractAndSort(image))

    # gray = cv2.fastNlMeansDenoising(gray, h=10)
    # binary = cv2.threshold(
    #     gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
    #                         threshold=50, minLineLength=100, maxLineGap=10)
    # image = rotate(image, lines)
    # recognisedPhrases.append(extractAndSort(image))
    # separator = "|"
    # extractedWordsFromFirstList = []
    # extractedWordsFromSecondList = []
    # extractedWordsFromThirdList = []

    # for phrase in recognisedPhrases[1]:
    #     phraseWords = phrase.split(' ')
    #     for word in phraseWords:
    #         extractedWordsFromSecondList.append((word, None))

    # for phrase in recognisedPhrases[2]:
    #     phraseWords = phrase.split(' ')
    #     for word in phraseWords:
    #         extractedWordsFromThirdList.append((word, None))

    # for phrase in recognisedPhrases[0]:
    #     extractedWordsFromFirstList = phrase.split(' ')
    #     for i, word in enumerate(extractedWordsFromFirstList):
    #         if (word.HasNoMeaning):
    #             # replace the word with the corresponding meaningful word of list2 or list3...

    # x = get_best_words(recognisedPhrases)
    # new = []
    # for phrase in recognisedPhrases[0]:
    #     extractedWordsFromFirstList = phrase.split(' ')
    #     for i, word in enumerate(extractedWordsFromFirstList):
    #         lastwordtoken = None
    #         # nextwordtoken = None
    #         lastlastwordtoken = None
    #         # nextnextwordtoken = None
    #         if (i > 0):
    #             lastword = extractedWordsFromFirstList[i-1]
    #             lastwordtoken = nlp(lastword)[0]
    #         # if (i < len(extractedWordsFromFirstList)-1):
    #         #     nextword = extractedWordsFromFirstList[i+1]
    #         #     nextwordtoken = nlp(nextword)[0]

    #         if (i > 1):
    #             lastlastword = extractedWordsFromFirstList[i-2]
    #             lastlastwordtoken = nlp(lastlastword)[0]
    #         if (i < len(extractedWordsFromFirstList)-2):
    #             nextnextword = extractedWordsFromFirstList[i+2]
    #             nextnextwordtoken = nlp(nextnextword)[0]
    #         if (not nlp.vocab[word].is_stop):
    #             continue
    #         token = nlp(word)[0]  # create a token using the spacy model
    #         if not token.has_vector or token.is_oov:  # check if the word has a vector or not
    #             # replace the word with the corresponding meaningful word of list2 or list3...
    #             best_match = None
    #             best_similarity = 0
    #             bigList = extractedWordsFromSecondList + extractedWordsFromThirdList
    #             for j, (meaningfulWord, _) in enumerate(bigList):
    #                 if (j > 1 and j < len(bigList)-2):
    #                     if (lastwordtoken != None and nlp(bigList[j-1][0])[0].similarity(lastwordtoken) == 1):
    #                         if (lastlastwordtoken != None and nlp(bigList[j-2][0])[0].similarity(lastlastwordtoken) == 1):
    #                             similarity = nlp(meaningfulWord)[
    #                                 0].similarity(token)
    #                             if similarity >= best_similarity and similarity > 0.1:
    #                                 best_match = (meaningfulWord, j)
    #                                 best_similarity = similarity
    #                             else:
    #                                 best_match = ('', j)
    #             if best_match is not None:
    #                 extractedWordsFromFirstList[i] = best_match[0]

    #     extractedWordsFromFirstList = [
    #         (word, i) for i, word in enumerate(extractedWordsFromFirstList)]
    #     new.append(extractedWordsFromFirstList)
    # print(new)  # output: "hi am aziz"

    response = {'text': recognisedPhrases}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
