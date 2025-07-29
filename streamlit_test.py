import cv2
import numpy as np
import base64
from PIL import Image
import io
import google.generativeai as genai
from datetime import datetime
import mediapipe as mp
import streamlit as st

# Configure Gemini API
GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def process_hand_region(frame):
    if frame is None or frame.size == 0:
        return None, None
        
    try:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Create a mask for visualization
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if not results.multi_hand_landmarks:
            return mask, frame
        
        # Draw hand landmarks and create mask
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Create a polygon from hand landmarks
            points = []
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append([x, y])
            
            # Create convex hull around hand
            if points:  # Only proceed if we have points
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)

        # Additional image processing for better segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Combine MediaPipe and skin color masks
        combined_mask = cv2.bitwise_and(mask, skin_mask)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.GaussianBlur(combined_mask, (5,5), 50)
        
        return combined_mask, frame
        
    except Exception as e:
        print(f"Error in process_hand_region: {str(e)}")
        return None, None

def get_gemini_prediction(image, previous_prediction=None):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhanced prompt with context
        prompt = f"""
        Analyze this Indian Sign Language gesture image. Focus on:
        1. Hand shape and finger positions
        2. Orientation of the palm
        3. Any distinct patterns or configurations
        
        and only return the word you predict, nothing else please.
        
        """
        
        response = model.generate_content([prompt, image_pil])
        return response.text
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def cleanup_resources(cap=None):
    """Clean up camera and window resources"""
    print("Cleaning up resources...")
    if 'hands' in globals():
        hands.close()
    if cap is not None:
        cap.release()

def initialize_camera():
    """Initialize and configure the camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open camera")
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def process_frame(frame, roi_coords, previous_prediction):
    roi_top, roi_bottom, roi_left, roi_right = roi_coords
    
    # Mirror the frame for more intuitive interaction
    frame = cv2.flip(frame, 1)
    
    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    
    # Extract and process ROI
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    if roi is None or roi.size == 0:
        return frame, roi, None
        
    processed_mask, annotated_roi = process_hand_region(roi)
    
    # Only proceed with overlay if we have valid mask and ROI
    if processed_mask is not None and annotated_roi is not None:
        mask_indices = processed_mask > 0
        if np.any(mask_indices):
            display_roi = roi.copy()
            green_overlay = np.full_like(roi, [0, 255, 0])
            display_roi[mask_indices] = cv2.addWeighted(
                roi[mask_indices], 0.7,
                green_overlay[mask_indices], 0.3,
                0
            )
            frame[roi_top:roi_bottom, roi_left:roi_right] = display_roi
        else:
            frame[roi_top:roi_bottom, roi_left:roi_right] = roi
    
    # Add instructions and previous prediction to frame
    cv2.putText(frame, "Press 'capture' below to capture", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if previous_prediction:
        cv2.putText(frame, f"Previous Word: {previous_prediction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, roi, processed_mask

def save_capture(roi, prediction):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{prediction}.jpg"
    cv2.imwrite(filename, roi)
    print(f"\nPrediction: {prediction}")
    print(f"Frame saved as {filename}")

def main():
    st.title("Indian Sign Language Detection")
    st.write("Position your hand in the green rectangle and press 'Capture' to predict the gesture.")

    # Define ROI coordinates
    roi_coords = (80, 400, 200, 520)  # (top, bottom, left, right)
    previous_prediction = None
    cap = None

    try:
        cap = initialize_camera()
        frame_placeholder = st.empty()
        capture_button = st.button("Capture")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.error("Error: Failed to grab frame")
                break
                
            try:
                frame, roi, processed_mask = process_frame(frame, roi_coords, previous_prediction)
                
                # Display the frame in the Streamlit app
                frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                if capture_button and roi is not None and roi.size > 0:
                    prediction = get_gemini_prediction(roi, previous_prediction)
                    previous_prediction = prediction
                    save_capture(roi, prediction)
                    st.success(f"Prediction: {prediction}")
                    capture_button = False
                    
            except Exception as e:
                st.error(f"Error in main loop: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
    
    finally:
        cleanup_resources(cap)

if __name__ == "__main__":
    main()
