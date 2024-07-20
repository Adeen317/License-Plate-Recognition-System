from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pytesseract
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_license_plate(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (1200, 800))  # Increased image size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        inverted_image = 1 - (gray / 255.0)
        inverted_image_uint8 = (inverted_image * 255).astype(np.uint8)
        inverted_image_filtered = cv2.bilateralFilter(inverted_image_uint8, 13, 75, 75)

        th2 = cv2.adaptiveThreshold(inverted_image_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        edged = cv2.Canny(th2, 30, 200)
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.036 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            mask = np.zeros(inverted_image.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 1, -1)
            new_image = cv2.bitwise_and(inverted_image, inverted_image, mask=mask)

            (x, y) = np.where(mask == 1)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = inverted_image[topx:bottomx + 1, topy:bottomy + 1]

            # Save processed images
            original_filename = os.path.basename(image_path)
            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + original_filename)
            
            # Ensure the cropped image is saved correctly
            if cropped is not None:
                cv2.imwrite(processed_image_path, (cropped * 255).astype(np.uint8))
                print(f"Processed Image Saved: {processed_image_path}")  # Debugging print

            text = pytesseract.image_to_string((cropped * 255).astype(np.uint8),
                                               config='--psm 11 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            return text, original_filename, processed_image_path
        else:
            print("No license plate contour found.")  # Debugging print
            return "No license plate contour found.", None, None
    except Exception as e:
        print(f"Error processing license plate: {e}")  # Debugging print
        return "Error processing image.", None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            text, original_image, processed_image_path = process_license_plate(filepath)
            print(f"Original Image Path: {original_image}")  # Debugging print
            print(f"Processed Image Path: {processed_image_path}")  # Debugging print
            return render_template('index.html', text=text, original_image=original_image, cropped_image=processed_image_path)

    return render_template('index.html', text='', original_image='', cropped_image='')



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
