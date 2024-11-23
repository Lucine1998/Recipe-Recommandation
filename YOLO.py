import torch
import json
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Cache the model globally
CACHED_MODEL = None

class YOLOProcessor:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model = None
        self.results = None

    def load_model(self):
        """Load the YOLO model."""
        global CACHED_MODEL
        if CACHED_MODEL is None:
            print(f"Loading model from {self.weights_path} for the first time.")
            CACHED_MODEL = torch.hub.load("ultralytics/yolov5", "custom", path=self.weights_path, force_reload=True)
        else:
            print("Using cached model.")
        self.model = CACHED_MODEL

    def load_image(self, image):
        """Load the input image."""
        print(f"Loading image...")
        if isinstance(image, str):  # If the input is a path, load the image from file
            img = Image.open(image)
        elif isinstance(image, Image.Image):  # If it's already a PIL Image
            img = image
        elif isinstance(image, np.ndarray):  # If it's a numpy array, convert it to PIL Image
            img = Image.fromarray(image)
        else:
            raise ValueError("Unsupported image type. Expected a file path, PIL Image, or numpy array.")

        return np.array(img)


    def perform_inference(self, image):
        """Perform inference on the input image."""
        print("Performing inference...")
        self.results = self.model(image)

    def display_results(self, image):
        """Display and annotate the results on the image."""
        annotated_image = image.copy()

        for *box, conf, cls in self.results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.results.names[int(cls)]} {conf:.2f}"

            # Draw the bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate label position
            y_label = y1 - 10 if y1 - 10 > 10 else y1 + 20

            # Add a filled rectangle for better label readability
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_image, (x1, y_label - text_height - 5), (x1 + text_width, y_label + baseline - 5),
                          (0, 255, 0), thickness=-1)

            # Draw the label text
            cv2.putText(annotated_image, label, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Convert BGR to RGB for displaying with matplotlib
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Display the image
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()

        # Save the annotated image
        cv2.imwrite('result.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def generate_class_counts_json(self, output_file="class_counts.json"):
        """Generate a JSON file with counts of each detected class."""
        # Extract all detected class indices
        class_indices = [int(cls) for cls in self.results.xyxy[0][:, 5]]

        # Map indices to class names
        class_names = [self.results.names[idx] for idx in class_indices]

        # Count occurrences of each class
        class_counts = Counter(class_names)

        # Convert to dictionary format for JSON
        class_counts_dict = dict(class_counts)

        # Save the class counts to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(class_counts_dict, json_file, indent=4)

        print(f"Class counts JSON file saved to {output_file}")
        return class_counts_dict

    def process(self, image_path):
        """Complete pipeline to process the image."""
        self.load_model()
        image = self.load_image(image_path)
        self.perform_inference(image)
        self.display_results(image)
        json_counts = self.generate_class_counts_json()
        
        return str(json_counts)

if __name__ == "__main__":
    # Define paths
    weights_path = "best.pt"
    image_path = "test.jpg"

    # Create YOLOProcessor instance and process the image
    yolo_processor = YOLOProcessor(weights_path, image_path)
    yolo_processor.process()