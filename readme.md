# License Plate Recognition System

This repository contains the implementation of a license plate recognition system using PyTesseract, OpenCV, and Flask for backend development. The frontend is developed using HTML and CSS.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Introduction

The License Plate Recognition System aims to automatically detect and recognize license plate numbers from images. It leverages optical character recognition (OCR) to process images and extract relevant information, making it suitable for applications such as automated parking systems and security.

## Features

- **Automatic License Plate Detection**: Identify and extract license plates from input images.
- **Optical Character Recognition**: Use PyTesseract for extracting text from the detected license plates.
- **User-Friendly Interface**: Upload images and view results through a simple web interface.
- **Extensible and Modular Design**: Easily extend the system with additional features.

![L P N R S-Work-ezgif com-optimize](https://github.com/user-attachments/assets/15a404d3-a9d8-49cc-8674-724c4cfcaf61)

## Technologies Used

- **Backend**: Flask, PyTesseract, OpenCV
- **Frontend**: HTML, CSS
- **Other Tools**: Python, NumPy

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/Adeen317/License-Plate-Recognition-System.git
   cd license-plate-recognition
   ```

2. **Set up a virtual environment:**
   ```
   python -m venv venv
   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR:**
   - **Windows**: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: Install via Homebrew:
     ```bash
     brew install tesseract
     ```
   - **Linux**: Install via package manager:
     ```bash
     sudo apt-get install tesseract-ocr
     ```

## Usage

1. **Start the Flask application:**
   ```
   flask run
   ```

2. **Open your web browser and go to:**
   ```
   http://localhost:5000
   ```


## Project Structure

```
license-plate-recognition/
│
├── app.py                  # Main application file
├── static/                 # Static files (CSS, images)
├── templates/              # HTML templates
└── requirements.txt        # List of dependencies
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push to your fork and submit a pull request.



