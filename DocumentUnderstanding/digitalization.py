import cv2
import numpy as np
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import xlsxwriter
import math
import json
from scipy import spatial
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

import classes
from classes import Bewerber, Erziehungsberechtigter

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def save_and_show_image(name, image):
    cv2.imwrite(f"temp/{name}.jpg", image)
    display(f"temp/{name}.jpg")


def detect_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangle_coordinates = []

    # Find the largest contour (which approximates the document area)
    largest_contour_index = np.argmax([cv2.contourArea(cnt) for cnt in contours])
    largest_contour = contours[largest_contour_index]

    # Get the bounding box coordinates of the largest contour
    x_largest, y_largest, w_largest, h_largest = cv2.boundingRect(largest_contour)

    # Crop the original image using the bounding box of the largest contour
    cropped_image = image[y_largest:y_largest + h_largest, x_largest:x_largest + w_largest]
    # save_and_show_image("cropped.jpg", cropped_image)
    # Find contours again within the cropped image
    gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, threshold_cropped = cv2.threshold(gray_cropped, 127, 255, cv2.THRESH_BINARY)
    contours_cropped, hierarchy_cropped = cv2.findContours(threshold_cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour index again within the cropped contours
    largest_contour_index_cropped = np.argmax([cv2.contourArea(cnt) for cnt in contours_cropped])

    # Iterate through all contours
    for i, contour in enumerate(contours_cropped):
        # Check if the contour is a direct child of the largest contour
        if hierarchy_cropped[0][i][3] == largest_contour_index_cropped:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # Define the minimum percentage threshold for considering contours
            min_percentage_threshold = 0.05  # You can adjust this value as needed

            # Compute the percentage of the document area occupied by the contour
            contour_percentage = (area / (w_largest * h_largest)) * 100

            # Check if the contour occupies less than the specified percentage of the document area
            if contour_percentage >= min_percentage_threshold:
                # Get the bounding box coordinates of the current contour
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
                rectangle_coordinates.append((x_rect, y_rect, w_rect, h_rect))

    return rectangle_coordinates, cropped_image




def find_labels(censored_image):
    # Grayscale and Otsu's thresholding
    gray = cv2.cvtColor(censored_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilate with vertical kernel to connect characters
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    dilate_x = cv2.dilate(thresh, kernel_x, iterations=4)
    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    dilate = cv2.dilate(dilate_x, kernel_y, iterations=3)

    save_and_show_image("dilated", dilate)

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine average contour area
    average_area = sum(cv2.contourArea(c) for c in contours) / len(contours)

    # Filter contours based on area
    accepted_contours = []
    for c in contours:
        if average_area * 0.15 < cv2.contourArea(c) < average_area * 4:
            x, y, w, h = cv2.boundingRect(c)
            accepted_contours.append((x, y, w, h))
            cv2.rectangle(censored_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    save_and_show_image("dilated_with_rectangles", censored_image)

    return accepted_contours


def get_labels(censored_image, coords):
    # Sort coordinates based on their position (top-left to bottom-right)
    # sorted_coords = coords # sorted(coords, key=lambda x: (x[1], x[0]))  # Sort by y-coordinate, then by x-coordinate

    data = {'text': [], 'contour_coordinates': []}

    for rectangle in coords:
        # Extract OCR result inside the rectangle
        x, y, w, h = rectangle
        print(f"Bounding rectangle of label: x={x}, y={y}, w={w}, h={h}")

        region_image = censored_image[y:y + h, x:x + w]
        region_text = pytesseract.image_to_string(region_image, lang='deu')
        cv2.rectangle(censored_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness 2
        if region_text != "":
            # Append the text and its coordinates to the dictionary
            data['text'].append(region_text)
            data['contour_coordinates'].append((x, y, w, h))

    save_and_show_image("get_labels_region_image", censored_image)
    return data

def get_labels_in_document(censored_image):
    coords = find_labels(censored_image)
    labels = get_labels(censored_image, coords)
    return labels
def extract_information(image, rect):
    x, y, w, h = rect
    print(f"x: {x}, y: {y}, w: {w}, h: {h}")
    region_image = image[y:y+h, x:x+w]
    region_text = pytesseract.image_to_string(region_image, lang='deu')
    return region_text.strip()


def find_closest_word(rect, words_info):
    rect_center = Point((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)

    word_centers = []

    for word_coordinates in words_info['contour_coordinates']:
        x, y, w, h = word_coordinates
        word_contour = np.array([(x, y), (x, y + h), (x + w, y + h), (x + w, y)], dtype=np.int32)
        M = cv2.moments(word_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            word_center = Point(cx, cy)
            word_centers.append(word_center)

    # Create a MultiPoint object from the word centers
    words_centers = MultiPoint(word_centers)

    # Find the nearest word center to the rectangle center
    nearest_point = nearest_points(rect_center, words_centers)[1]

    # Define a threshold distance for considering points as identical
    threshold_distance = 1.0  # Adjust this value according to your requirements

    # Find the index of the nearest word center
    nearest_index = word_centers.index(nearest_point)

    # Retrieve the closest word using the index
    closest_word = words_info['text'][nearest_index]

    # Calculate the distance between the rectangle center and the closest word center
    distance = rect_center.distance(nearest_point)

    print(f"Closest word: {closest_word}, Distance: {distance}")
    return closest_word, nearest_point



def process_rectangles(rectangle_coordinates, words_info, image):
    linked_data = {}

    for rectangle in rectangle_coordinates:
        # Find the center of the rectangle
        rect_center = [(rectangle[0] + rectangle[0] + rectangle[2]) / 2,
                        (rectangle[1] + rectangle[1] + rectangle[3]) / 2]

        # Find the center of the closest word to the rectangle
        closest_word, closest_word_center = find_closest_word(rectangle, words_info)

        # Extract OCR result inside the rectangle
        x, y, w, h = rectangle
        region_image = image[y:y+h, x:x+w]
        region_text = pytesseract.image_to_string(region_image, lang='deu')

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        if closest_word_center is not None:
            # Draw a line between the center of the rectangle and the center of the closest word
            cv2.line(image, (int(rect_center[0]), int(rect_center[1])),
                        (int(closest_word_center.x), int(closest_word_center.y)),
                            (0, 0, 255), 2)  # Red line

            # Draw a big blue dot at the center of the rectangle and the center of the closest word
            cv2.circle(image, (int(rect_center[0]), int(rect_center[1])), 10, (255, 0, 0), -1)  # Blue dot
            cv2.circle(image, (int(closest_word_center.x), int(closest_word_center.y)), 10, (255, 0, 0), -1)  # Blue dot

        # Link the data from the rectangle to the corresponding word
        linked_data[region_text.strip()] = closest_word

    save_and_show_image("region_image", image)
    return linked_data

def process_image(image_path):
    image = cv2.imread(image_path)

    rectangle_coordinates, cropped_image = detect_boxes(image)
    censored_image = cropped_image.copy()
    # Draw rectangles over the identified regions
    for rect in rectangle_coordinates:
        x, y, w, h = rect
        cv2.rectangle(censored_image, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)

    # Display the modified image with rectangles covering the identified regions
    save_and_show_image("covered_rectangles", censored_image)

    words_info = get_labels_in_document(censored_image)

    linked_data = process_rectangles(rectangle_coordinates, words_info, cropped_image)
    bewerber = Bewerber.getBewerber(linked_data)
    # Serialize bewerber object to JSON
    bewerber_json = json.dumps(bewerber.__dict__)

    # Save JSON to the output directory
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.basename(image_path)}.json")
    with open(output_file, 'w') as f:
        f.write(bewerber_json)

    return bewerber_json

# Path to the directory to monitor
DATA_DIR = 'data'
OUTPUT_DIR = 'output'  # Define the output directory

def main():
    # Keep track of the files currently in the directory
    current_files = set()

    while True:
        # List all files in the directory
        files = set(os.listdir(DATA_DIR))

        # Find new files by comparing the current and new file sets
        new_files = files - current_files

        # Process each new file
        for file in new_files:
            file_path = os.path.join(DATA_DIR, file)
            # Process the new file
            bewerber_json = process_image(file_path)
            print(f'Processed file: {file_path}')
            print(f'Bewerber JSON: {bewerber_json}')

        # Update the current file set
        current_files = files

        # Wait for a while before checking again
        time.sleep(10)  # Adjust the interval as needed

if __name__ == "__main__":
    main()
