import keras_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

image = cv2.imread('content/text.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

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
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), average_angle, 1)
images = [cv2.warpAffine(image, rotation_matrix, (cols, rows))]
prediction_groups = pipeline.recognize(images)

# UNCOMMENT TO SEE FIGURE

# rows = 1
# columns = 2

# fig, axs = plt.subplots(ncols=columns, nrows=rows, figsize=(25, 15))
# axs = axs.flatten()

# for ax, image, predictions in zip(axs, images, prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image,
#                                     predictions=predictions,
#                                     ax=ax)

# plt.show()
# plt.waitforbuttonpress()


predicted_image_1 = prediction_groups[0]
predicted_image_1_sorted = sorted(predicted_image_1, key=lambda x: (
    x[1][0][1], x[1][0][0]))  # write output text to file in correct order

phrases = []
current_phrase = []

for i in range(len(predicted_image_1_sorted)):
    word = predicted_image_1_sorted[i][0]
    box = predicted_image_1_sorted[i][1]

    if i == 0:
        current_phrase.append(word)
    else:
        prev_box = predicted_image_1_sorted[i-1][1]

        # assume words within 5 pixels of each other are on the same line
        if abs(box[0][1] - prev_box[0][1]) <= 5:
            current_phrase.append(word)
        else:
            phrases.append(' '.join(current_phrase))
            current_phrase = [word]

    if i == len(predicted_image_1_sorted) - 1:
        phrases.append(' '.join(current_phrase))

print(phrases)
