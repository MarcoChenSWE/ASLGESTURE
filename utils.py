import streamlit as st

def display_gesture_chart(image_path):
    """
    Displays the gesture chart in the sidebar.

    Args:
        image_path (str): Path to the gesture chart image.
    """
    st.sidebar.markdown("### Gesture Reference")
    st.sidebar.image(image_path, caption="ASL Gesture Chart", use_column_width=True)
