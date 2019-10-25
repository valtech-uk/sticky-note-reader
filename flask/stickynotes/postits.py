import cv2 as cv
import numpy as np
import colorsys
from stickynotes import board
import pytesseract as pt
from matplotlib import pyplot as plt
from stickynotes import azure

def get_color(color):
    trello_colors = {
    'yellow': (242, 214, 0),
    'purple': (195, 119, 224),
    'blue': (0, 121, 191),
    'red': (235, 90, 70),
    'green': (97, 189, 79),
    'orange': (255, 159, 26),
    'black': (52, 69, 99),
    'sky': (0, 194, 224),
    'pink': (255, 120, 203),
    'lime': (81, 232, 151)
    }
    min_dist = 1
    min_col = None
    for name, rgb in trello_colors.items():
        new_dist = color_distance(rgb, color)
        if new_dist < min_dist:
            min_dist = new_dist
            min_col = name
    return min_col

def color_distance_euc(color1, color2):
    sum_square = 0
    for i in range(3):
        sum_square += (color1[i]-color2[i])**2
    return np.sqrt(sum_square)

def color_distance(color1, color2):
    color1 = colorsys.rgb_to_hsv(color1[0], color1[1], color1[2])
    color2 = colorsys.rgb_to_hsv(color2[0], color2[1], color2[2])
    return np.abs(color1[0]-color2[0])

def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    right = int(rightMin + (valueScaled * rightSpan))
    return max(0, min(right, 255))

def threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    flat = np.reshape(gray.astype(np.float32), (-1, 1))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    comp, labels, centres = cv.kmeans(flat, 3, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    max_val = 0
    for i in range(len(centres)):
        max_val = max(max_val, centres[i])
    mid  = (sum(centres) - max_val) / 2
    #mid = (centres[0] + centres[1]) / 2
    ret,thresh = cv.threshold(gray,mid,255,cv.THRESH_BINARY)
    return thresh

def contrast(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    flat = np.reshape(gray.astype(np.float32), (-1, 1))
    minpix, maxpix = np.percentile(flat, [3, 97])
    
    mapper = lambda x: translate(x, minpix, maxpix, 0, 255)
    vmap = np.vectorize(mapper)
    cont = vmap(gray)
    return cont.astype(np.uint8)

def image_to_bytes(image):
    ret, image = cv.imencode(".jpg", image)
    bytesimg = image.tobytes()
    return bytesimg

def note_text(image):
    readable_image = threshold(image)
    cv.imwrite('output_images/image{}x{}.jpg'.format(image.shape[0], image.shape[1]), readable_image)

    text = pt.image_to_string(readable_image)
    return text

def note_color(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return get_color(avg_color)

def postit_card(name, image):
    card = {'name': name, 'label': note_color(image)}
    return card

def postit_group(text_detector, name, images):
    group = {'name': name, 'cards':[]}
    byteimages = []
    for image in images:
        byteimages.append(image_to_bytes(image))
    text = text_detector.batch_infer(byteimages)
    for i in range(len(images)):
        group['cards'].append(postit_card(text[i], images[i]))
    return group

def postit_board(postit_detector, text_detector, key, token, name, image, username, email):
    images = postit_detector.infer_images(image)
    group = postit_group(text_detector, 'group1', images)
    newBoard = board.Board(key, token, name, [group])
    newBoard.make()
    newBoard.invite(username, email)
    return newBoard.url

def postit_textonly(postit_detector, text_detector, image):
    images = postit_detector.infer_images(image)
    byteimages = []
    for image in images:
        byteimages.append(image_to_bytes(image))
    text = text_detector.batch_infer(byteimages)
    return text

