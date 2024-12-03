# without hand landmarking
# import streamlit as st
# import mediapipe as mp
# import cv2
# import os
# import time
# from queue import Queue

# # Import necessary components from MediaPipe
# BaseOptions = mp.tasks.BaseOptions
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Correct path to the Gesture Recognizer model file
# model_path = './model/model/gesture_recognizer.task'

# # Check if file exists
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # Queue to share results between the callback and main thread
# gesture_queue = Queue()

# # Callback function to process results and add them to the queue
# def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     results = []  # Collect gesture results
#     if result.gestures:
#         for hand_gestures in result.gestures:
#             for gesture in hand_gestures:
#                 results.append(f"Gesture: **{gesture.category_name}**, Confidence: **{gesture.score:.2f}**")
#     else:
#         results.append("No gestures detected.")
#     gesture_queue.put(results)

# # Configure the Gesture Recognizer
# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result
# )

# # Custom App Header
# st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Gesture Recognition App 🚀</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; color: grey;'>Recognize hand gestures in real time with MediaPipe and Streamlit</p>", unsafe_allow_html=True)

# # Sidebar for User Controls
# st.sidebar.title("Control Panel")
# run_app = st.sidebar.button("Start Gesture Recognition")
# st.sidebar.write("Toggle the button above to start the app.")

# # Placeholder for video feed and results
# video_placeholder = st.empty()  # Placeholder for the video feed
# result_placeholder = st.empty()  # Placeholder for gesture results

# # Footer with branding
# st.sidebar.markdown(
#     "<hr><p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True
# )

# if run_app:
#     st.markdown("<h2 style='text-align: center;'>Processing Video Feed...</h2>", unsafe_allow_html=True)
#     cap = cv2.VideoCapture(0)

#     # Initialize a monotonically increasing timestamp
#     start_time = time.time()

#     with GestureRecognizer.create_from_options(options) as recognizer:
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 st.warning("No frames available from the video feed.")
#                 break

#             # Compute the current timestamp in milliseconds
#             current_time_ms = int((time.time() - start_time) * 1000)

#             # Convert frame to a MediaPipe Image
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

#             # Perform gesture recognition asynchronously
#             recognizer.recognize_async(mp_image, current_time_ms)

#             # Display the frame in Streamlit
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             video_placeholder.image(frame_rgb, channels="RGB", caption="Gesture Recognition", use_column_width=True)

#             # Retrieve and display gesture results from the queue
#             while not gesture_queue.empty():
#                 results = gesture_queue.get()
#                 result_placeholder.markdown(
#                     "<h3 style='text-align: center; color: #FF5722;'>Detected Gestures</h3>",
#                     unsafe_allow_html=True,
#                 )
#                 result_placeholder.markdown(
#                     "<ul>" + "".join([f"<li>{result}</li>" for result in results]) + "</ul>",
#                     unsafe_allow_html=True,
#                 )

#     cap.release()














# with hand landmark
import streamlit as st
import mediapipe as mp
import cv2
import os
import time
from queue import Queue
from utils import display_gesture_chart

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Import necessary components from MediaPipe Gesture Recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Correct path to the Gesture Recognizer model file
model_path = './model/model/gesture_recognizer.task'

# Check if file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Queue to share results between the callback and main thread
gesture_queue = Queue()

# Callback function to process gesture results and add them to the queue
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    results = []  # Collect gesture results
    if result.gestures:
        for hand_gestures in result.gestures:
            for gesture in hand_gestures:
                results.append(f"{gesture.category_name} (Confidence: {gesture.score:.2f})")  # Include confidence
    else:
        results.append("No gestures detected.")
    gesture_queue.put(results)

# Configure the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize session state for saving gestures
if "recognized_gestures" not in st.session_state:
    st.session_state.recognized_gestures = []

# Custom App Header
st.markdown(
    """
    <style>
    .header {text-align: center; color: #4CAF50; margin-top: -50px;}
    .description {text-align: center; color: grey; font-size: 16px;}
    </style>
    <h1 class="header">Gesture & Hand Landmark Detection 🚀</h1>
    <p class="description">Recognize and save hand gestures in real time with MediaPipe.</p>
    """,
    unsafe_allow_html=True,
)

# Sidebar for User Controls
st.sidebar.title("Control Panel")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Display gesture chart in the sidebar
gesture_chart_path = "./gestureReference.png"  # Update this with the actual path to the image
display_gesture_chart(gesture_chart_path)

max_num_hands = st.sidebar.slider("Max Number of Hands", 1, 2, 1)
skip_frames = st.sidebar.slider("Process Every Nth Frame", 1, 10, 5)
resolution = st.sidebar.selectbox("Frame Resolution", ["320x240", "640x480"], index=0)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Start and Stop buttons
if "run_app" not in st.session_state:
    st.session_state.run_app = False

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ Start"):
    st.session_state.run_app = True

if col2.button("⏹ Stop"):
    st.session_state.run_app = False

# Clear history button
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.recognized_gestures = []

# Layout with columns: Live camera feed on the left, gesture log box on the right
col_feed, col_log = st.columns([5, 2])

with col_feed:
    st.markdown("### Live Camera Feed")
    video_placeholder = st.empty()  # Placeholder for the video feed

with col_log:
    st.markdown("### Gesture Log")
    current_gesture_box = st.empty()  # Box to display the most recent gesture dynamically
    st.markdown("### Gesture History")
    gesture_history_box = st.empty()  # Box to display all recognized gestures dynamically

# Footer with branding
st.sidebar.markdown(
    """
    <style>
    .footer {text-align: center; font-size: 12px; color: grey; margin-top: 20px;}
    </style>
    <p class="footer">Made by Marco Chen, William Taka, Rigoberto Ponce using Streamlit, MediaPipe & OpenCV</p>
    """,
    unsafe_allow_html=True,
)

if st.session_state.run_app:
    cap = cv2.VideoCapture(0)

    # Parse resolution
    res_width, res_height = map(int, resolution.split("x"))

    # Initialize a monotonically increasing timestamp
    start_time = time.time()

    with GestureRecognizer.create_from_options(options) as recognizer, mp_hands.Hands(
        max_num_hands=max_num_hands,
        model_complexity=1,  # Simplified model for performance 0 for faster run
        min_detection_confidence=0.5, #0.3
        min_tracking_confidence=0.5  #0.3
    ) as hands:
        frame_count = 0
        while st.session_state.run_app and cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.warning("No frames available from the video feed.")
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Flip and resize frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (res_width, res_height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform hand landmark detection
            hand_results = hands.process(frame_rgb)

            # Perform gesture recognition
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            current_time_ms = int((time.time() - start_time) * 1000)
            recognizer.recognize_async(mp_image, current_time_ms)

            # Draw hand landmarks on the frame
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Retrieve and display gesture results from the queue
            while not gesture_queue.empty():
                results = gesture_queue.get()
                if results:
                    new_gesture = results[-1]

                    # Extract label and confidence safely
                    if " (Confidence: " in new_gesture:
                        label, confidence = new_gesture.split(" (Confidence: ")
                        confidence = confidence.rstrip(")")  # Remove the trailing parenthesis
                    else:
                        label = new_gesture
                        confidence = "N/A"

                    # Add new gesture to history only if it's not already logged
                    if label.isalpha() and new_gesture not in st.session_state.recognized_gestures:
                        st.session_state.recognized_gestures.append(new_gesture)

                        # Update current gesture display
                        current_gesture_box.markdown(
                            f"<h4 style='text-align: center; color: #4CAF50;'>Gesture: {label}<br>Confidence: {confidence}</h4>",
                            unsafe_allow_html=True,
                        )

                        # Update gesture history display
                        gesture_history_box.text_area(
                            "Gesture History:",
                            value="\n".join(reversed(st.session_state.recognized_gestures)),
                            height=300,
                            disabled=True,
                        )

            # Display the frame with hand landmarks and gesture results
            video_placeholder.image(frame, channels="BGR", caption="Gesture & Hand Landmark Detection", use_column_width=True)

    cap.release()



















