import keras_ocr
import cv2
import matplotlib.pyplot as plt

# Create a Keras OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Load the image
image_path = "C:\\Users\\thava\\OneDrive\\Pictures\\Screenshots\\Screenshot 2024-08-25 003356.png"
image = keras_ocr.tools.read(image_path)

# Use the Keras OCR pipeline to detect and recognize text
words = pipeline.recognize([image])[0]

# Output the recognized text and their corresponding bounding boxes
print("Extracted Text:")
for word in words:
    text, box = word
    print(f'Text: {text}, Bounding Box: {box}')

# Optional: Display the original image with detected text boxes
plt.figure(figsize=(10, 6))
plt.imshow(image)
for text, box in words:
    # Draw the bounding box around detected text
    plt.plot([box[0][0], box[1][0], box[2][0], box[3][0], box[0][0]], 
             [box[0][1], box[1][1], box[2][1], box[3][1], box[0][1]], 
             color='red')
    plt.text(box[0][0], box[0][1], text, fontsize=12, color='blue')

plt.title("Detected Text with Keras OCR")
plt.axis('off')
plt.show()
