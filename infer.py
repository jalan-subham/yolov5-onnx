import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import argparse

# Load YOLOv5 ONNX model
onnx_model_path = 'yolov5m.onnx'
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Function to preprocess the input image
def preprocess_image(image_path, input_size=(640, 640)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size, Image.BICUBIC)
    img = np.array(img).astype(np.float32)  # Convert to numpy array as float32
    img /= 255.0  # Normalize to 0-1
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension (NCHW)

    # Convert to float16
    img = img.astype(np.float16)  # Convert to float16 as expected by the model
    return img

# Post-process the predictions (Non-Maximum Suppression and bounding box formatting)
def post_process(predictions, conf_threshold=0.25, iou_threshold=0.45):
    # YOLO model outputs: (batch_size, grid, grid, anchors, [x, y, w, h, conf, class_probs])
    boxes = predictions[..., :4]  # Bounding box coordinates (x, y, w, h)
    conf = predictions[..., 4:5]  # Confidence score (objectness score)
    class_probs = predictions[..., 5:]  # Class probabilities (for all classes)

    # Calculate confidence score (confidence * class probabilities)
    scores = conf * class_probs  # Shape: (batch_size, grid, grid, anchors, num_classes)

    # Reshape predictions to (batch_size * grid * grid * anchors, num_classes)
    scores = scores.reshape(-1, scores.shape[-1])  # Flatten the scores to make them easier to handle
    boxes = boxes.reshape(-1, 4)  # Flatten the bounding box coordinates
    conf = conf.reshape(-1, 1)  # Flatten the confidence scores

    # Apply a threshold to filter out low-confidence boxes
    mask = scores.max(axis=1) > conf_threshold  # Confidence threshold
    boxes = boxes[mask]
    scores = scores[mask]
    conf = conf[mask]

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.max(axis=1).tolist(), conf_threshold, iou_threshold)

    # Collect results
    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            x1, y1, x2, y2 = boxes[idx]
            score = scores[idx].max()  # Use the max class score for final confidence
            results.append((x1, y1, x2, y2, score))
    return results

# Main function to run the inference
def run_inference(image_path):
    # Load and preprocess the image
    input_image = preprocess_image(image_path)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image})

    # Process the output (YOLOv5 outputs: predictions)
    predictions = outputs[0]  # The model outputs are in the first element

    # Post-process the results
    results = post_process(predictions[0])

    # Draw the bounding boxes on the image
    image = cv2.imread(image_path)
    for (x1, y1, x2, y2, score) in results:
        # Convert coordinates back to pixel values
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(image, f'{score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow('Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(description="Run YOLOv5 Inference on an Image")
    parser.add_argument('image', type=str, help="Path to the input image file")
    args = parser.parse_args()

    # Run inference on the provided image
    run_inference(args.image)
