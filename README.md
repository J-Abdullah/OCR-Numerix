# OCR System

This project demonstrates an approach to building an OCR (Optical Character Recognition) system using YOLO for object detection and Tesseract for text recognition. The models are trained on a very small dataset, and the OCR function is not very accurate. This project is intended to show the approach and will be improved in the future.

## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/J-Abdullah/OCR-Numerix.git
   cd ocr_system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that Tesseract OCR is installed on your system. You can download it from [here](https://github.com/tesseract-ocr/tesseract).

4. Update the `pytesseract.pytesseract.tesseract_cmd` path in the `ocr_system.py` file to point to the Tesseract executable on your system:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'
   ```

## Usage

1. Place the image you want to process in the project directory.

2. Run the `ocr_system.py` script:
   ```bash
   python ocr_system.py
   ```

3. The script will process the image, perform OCR, and save the results to `ocr_system_results.csv`. It will also print the filtered results to the console.

## Example

To run the system on a sample image:

1. Place your image in the project directory and update the `image_path` variable in the `ocr_system.py` file:
   ```python
   image_path = 'sample.jpeg'
   ```

2. Run the script:
   ```bash
   python ocr_system.py
   ```

## Limitations

- The models are trained on a very small dataset, so the accuracy may not be high.
- The OCR function is not very accurate and may not work well on all types of documents.
- This project is intended to show the approach and will be improved in the future.

## Future Improvements

- Train the models on a larger and more diverse dataset to improve accuracy.
- Enhance the OCR function to handle different types of documents and text more effectively.
- Implement automatic document type detection.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Steps to Add the `README.md` to Your Project

1. **Create a new file named `README.md` in your project directory**
2. **Copy and paste the above content into the `README.md` file**
3. **Save the file**

