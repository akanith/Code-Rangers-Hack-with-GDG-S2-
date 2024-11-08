import cv2
import pytesseract
import matplotlib.pyplot as plt

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load and preprocess the image
image_path = "C:\\Users\\thava\\OneDrive\\Pictures\\Temps\\WhatsApp Image 2024-10-18 at 19.47.24_3bdb8332.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding for better results on uneven lighting
adaptive_threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

# Optional: Dilation to enhance text features
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(adaptive_threshold, kernel, iterations=1)

# OCR with Pytesseract
custom_config = r'--oem 3 --psm 6'  # Default LSTM mode, assume a uniform block of text
extracted_text = pytesseract.image_to_string(dilated, config=custom_config)

# Output extracted text
print("Extracted Text:")
print(extracted_text)

# Optional: Display the original and processed images for comparison
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Processed Image")
plt.imshow(dilated, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
