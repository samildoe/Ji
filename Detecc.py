import cv2
import time
import csv
import numpy as np
import os

# --- Configuration ---
# Choose your target: 'jetson' or 'pi'
TARGET_DEVICE = 'jetson' 

# Model paths (update these after conversion)
MODEL_PATH_JETSON = 'best.engine'
MODEL_PATH_PI = 'best_quant.tflite'
MODEL_INPUT_SHAPE = (640, 640) # The input size your model expects

# Camera configuration (0 for USB cam, or RTSP URL for IP cam)
CAMERA_SOURCE = 0 
# CAMERA_SOURCE = "rtsp://user:pass@192.168.1.10:554/stream1"

# Product configuration (Price per Kilogram or per item)
PRODUCT_PRICES = {
    'rice': 2.50,    # Price per KG
    'lentils': 3.10, # Price per KG
    'flour': 1.80,   # Price per KG
    'pasta': 2.00    # Price per item
}
# Define which products are priced by weight
WEIGHT_BASED_PRODUCTS = {'rice', 'lentils', 'flour'}

# Logging configuration
LOG_FILE = 'detection_log.csv'

# --- Placeholder for Model Loading and Inference ---
# You will need to implement these functions based on your target device
# This is a conceptual guide.

class Model:
    def __init__(self, model_path):
        """
        Loads the model based on the target device.
        For Jetson: Use tensorrt and pycuda.
        For Pi: Use tflite_runtime.interpreter.
        """
        self.model_path = model_path
        if TARGET_DEVICE == 'jetson':
            # Example for Jetson (requires tensorrt, pycuda)
            # import tensorrt as trt
            # logger = trt.Logger(trt.Logger.WARNING)
            # with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            #     self.engine = runtime.deserialize_cuda_engine(f.read())
            # self.context = self.engine.create_execution_context()
            # ... more setup for bindings, inputs, outputs ...
            print("Jetson model loading logic goes here.")
            pass
        elif TARGET_DEVICE == 'pi':
            # Example for Raspberry Pi (requires tflite_runtime)
            # from tflite_runtime.interpreter import Interpreter
            # self.interpreter = Interpreter(model_path=model_path)
            # self.interpreter.allocate_tensors()
            # self.input_details = self.interpreter.get_input_details()
            # self.output_details = self.interpreter.get_output_details()
            print("Raspberry Pi TFLite model loading logic goes here.")
            pass
        else:
            raise ValueError("Invalid TARGET_DEVICE specified")

    def infer(self, image):
        """
        Performs inference on a single image.
        Returns a list of detections: [[x1, y1, x2, y2, class_id, score], ...]
        """
        # 1. Preprocess the image (resize, normalize, etc.)
        #    - Resize to MODEL_INPUT_SHAPE
        #    - Convert to float32
        #    - Normalize to [0, 1]
        #    - Add batch dimension
        
        # 2. Run inference using self.context (Jetson) or self.interpreter (Pi)
        
        # 3. Post-process the output to get bounding boxes, scores, and class IDs.
        #    - This involves parsing the raw model output and applying Non-Max Suppression.
        
        print("Inference and post-processing logic goes here.")
        # This is a placeholder for the actual output
        # In a real implementation, this would be populated by the model
        if np.random.rand() > 0.5: # Simulate a detection
             return [[100, 150, 300, 400, 0, 0.92]] # [x1, y1, x2, y2, class_id (rice), score]
        return []


def get_weight_from_sensor():
    """
    Placeholder function to get weight from a connected sensor (e.g., HX711).
    For now, it returns a default value.
    """
    # In a real implementation, you would interface with the GPIO pins
    # using a library like 'hx711'
    print("Reading weight from sensor (using default 1.0 kg for now).")
    return 1.0 # Default to 1 kg for weight-based items

def setup_log_file():
    """Creates the log file with a header if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Product Name', 'Weight (kg/item)', 'Calculated Price'])

def log_detection(product_name, weight, price):
    """Appends a single detection record to the CSV log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, product_name, f"{weight:.2f}", f"{price:.2f}"])

def main():
    # --- Initialization ---
    # You would need a mapping from class_id (integer) to class_name (string)
    class_names = ['rice', 'lentils', 'flour', 'pasta'] # Should match your training data

    # Load the model (uncomment when Model class is implemented)
    # model_path = MODEL_PATH_JETSON if TARGET_DEVICE == 'jetson' else MODEL_PATH_PI
    # model = Model(model_path)
    
    # Setup video capture
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open camera source '{CAMERA_SOURCE}'")
        return
        
    # Setup logging
    setup_log_file()
    print("System initialized. Starting real-time detection...")

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # --- Inference ---
        # detections = model.infer(frame) # Uncomment when model is ready
        detections = Model(None).infer(frame) # Using placeholder for now
        
        # --- Process Detections ---
        for det in detections:
            x1, y1, x2, y2, class_id, score = map(int, det[:5]) + [det[5]]
            
            if score < 0.5: # Confidence threshold
                continue

            product_name = class_names[class_id]
            
            # Calculate Price
            price_per_unit = PRODUCT_PRICES.get(product_name, 0)
            weight = 1.0 # Default for item-based pricing
            
            if product_name in WEIGHT_BASED_PRODUCTS:
                weight = get_weight_from_sensor()
            
            final_price = price_per_unit * weight
            
            # Log the detection
            log_detection(product_name, weight, final_price)

            # --- Visualization ---
            label = f"{product_name}: ${final_price:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Grocery Detection System', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("System shut down.")

if __name__ == '__main__':
    main()
              
