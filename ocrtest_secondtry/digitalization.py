import cv2
import numpy as np
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt

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
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangle_coordinates = []

    # Find the largest contour (which approximates the document area)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)
    save_and_show_image("largest_contour.jpg", image)

    # Define the minimum percentage threshold for considering contours
    min_percentage_threshold = 0.05  # You can adjust this value as needed

    # Calculate the area of the document
    document_area = cv2.contourArea(largest_contour)

    # Get the bounding box coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Iterate through all contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        # Compute the percentage of the document area occupied by the contour
        contour_percentage = (area / document_area) * 100

        # Check if the contour occupies less than the specified percentage of the document area
        if contour_percentage < min_percentage_threshold:
            continue

        # Get the bounding box coordinates of the current contour
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

        # Check if the rectangle falls inside the largest contour
        if x <= x_rect and y <= y_rect and x + w >= x_rect + w_rect and y + h >= y_rect + h_rect:
            rectangle_coordinates.append((x_rect, y_rect, w_rect, h_rect))

    return rectangle_coordinates


def extract_information(image, rect):
    x, y, w, h = rect
    print(f"x: {x}, y: {y}, w: {w}, h: {h}")
    region_image = image[y:y+h, x:x+w]
    region_text = pytesseract.image_to_string(region_image, lang='deu')
    return region_text.strip()


def find_closest_word(rect, words_info):
    rect_center = np.mean(rect, axis=0, dtype=int)
    closest_word = min(words_info, key=lambda word: np.linalg.norm(np.array([word['left'], word['top']]) - rect_center))
    return closest_word

def process_rectangles(rectangle_coordinates, words_info, image):
    linked_data = {}

    for rectangle in rectangle_coordinates:
        # Find the closest word to the rectangle
        closest_word = find_closest_word(rectangle, words_info)

        # Extract OCR result inside the rectangle
        x, y, w, h = rectangle
        print(f"Bounding rectangle: x={x}, y={y}, w={w}, h={h}")

        region_image = image[y:y+h, x:x+w]
        region_text = pytesseract.image_to_string(region_image, lang='deu')
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness 2


        # Link the data from the rectangle to the corresponding word
        linked_data[closest_word['text']] = region_text.strip()
    save_and_show_image("region_image.jpg", image)
    return linked_data

def process_image(image_path):
    image = cv2.imread(image_path)

    rectangle_coordinates = detect_boxes(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equalized_image = cv2.equalizeHist(threshold_image)

    data = pytesseract.image_to_data(equalized_image, lang='deu', output_type=pytesseract.Output.DICT)
    words_info = []

    for i in range(len(data['text'])):
        word = data['text'][i]
        left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        if word and width > 10 and height > 10:
            words_info.append({'text': word, 'left': left, 'top': top, 'width': width, 'height': height})

    linked_data = process_rectangles(rectangle_coordinates, words_info, image)

    df = pd.DataFrame(list(linked_data.items()), columns=['Word', 'Information'])
    df.to_excel('output.xlsx', index=False)

if __name__ == "__main__":
    process_image('data/test_image.jpg')
