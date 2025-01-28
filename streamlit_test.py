import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from datetime import datetime
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configure Gemini API
GOOGLE_API_KEY = 'AIzaSyDkLlJFajsNu1mJ2bek7dJGyAtFu05bYVI'
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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.previous_prediction = None
        self.roi_coords = (80, 400, 200, 520)  # (top, bottom, left, right)

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        
        # Process the frame
        frame, roi, _ = self.process_frame(frame, self.roi_coords, self.previous_prediction)
        
        # Display the frame
        return frame

    def process_frame(self, frame, roi_coords, previous_prediction):
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
        cv2.putText(frame, "Press 'Capture' to predict", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if previous_prediction:
            cv2.putText(frame, f"Previous Word: {previous_prediction}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, roi, processed_mask

def main():
    st.title("Indian Sign Language Detection")
    st.write("Position your hand in the green rectangle and press 'Capture' to predict the gesture.")

    # Initialize WebRTC streamer
    ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

    if ctx.video_transformer:
        if st.button("Capture"):
            roi = ctx.video_transformer.process_frame(
                ctx.video_transformer.frame,
                ctx.video_transformer.roi_coords,
                ctx.video_transformer.previous_prediction
            )[1]
            if roi is not None and roi.size > 0:
                prediction = get_gemini_prediction(roi, ctx.video_transformer.previous_prediction)
                ctx.video_transformer.previous_prediction = prediction
                st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
