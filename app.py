import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# Load the model
@st.cache_resource
def load_road_detection_model():
    """Load the pre-trained model."""
    return load_model('broken_road_model.h5')


def preprocess_image(image):
    """Preprocess the input image for prediction."""
    # Resize the image to match the model's expected input size (150x150)
    image = image.resize((150, 150))
    # Normalize pixel values to [0, 1]
    image_array = np.array(image) / 255.0
    # Reshape to match the model's input shape (1, 150, 150, 3)
    return image_array.reshape(1, 150, 150, 3)


class RoadDetectionTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        """Process the webcam feed frame."""
        img = frame.to_ndarray(format="bgr24")
        # Resize and normalize the frame for prediction
        img_resized = cv2.resize(img, (150, 150)) / 255.0
        img_input = img_resized.reshape(1, 150, 150, 3)

        # Get the prediction from the model
        prediction = self.model.predict(img_input)
        pred_label = np.argmax(prediction, axis=1)[0]
        label = "Good Road" if pred_label == 1 else "Bad Road"

        # Add the prediction label on the frame
        cv2.putText(
            img, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        return img


def main():
    """Main function to run the Streamlit app."""
    # Page Configuration
    st.set_page_config(page_title="Road Condition Detection App", layout="wide")

    # App Title and Header
    st.title("🚧 **Road Condition Detection App**")
    st.markdown("""
    Welcome to the **Road Condition Detection App**!  
    This app predicts whether a road is **Good** or **Bad** based on images or live webcam feed.  
    """)

    # Display Banner Image
    st.image("img.jpg", caption="Ensure Safer Roads!", use_container_width=True)

    # Load Model
    model = load_road_detection_model()

    # Tabs for different functionalities
    tabs = st.tabs(["📤 Image Upload", "📹 Webcam Feed"])

    with tabs[0]:
        st.markdown("### 📤 Upload an Image")
        uploaded_file = st.file_uploader("Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display uploaded image
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)

            # Preprocess Image
            preprocessed_image = preprocess_image(input_image)

            # Prediction Button
            st.markdown("### 🔍 Analyze Image")
            if st.button("PREDICT"):
                with st.spinner("Analyzing the image..."):
                    result = model.predict(preprocessed_image)
                    prediction = np.argmax(result, axis=1)[0]  # Binary classification

                # Display Results
                st.markdown("### 📋 Prediction Result")
                if prediction == 0:
                    st.error("❌ **Bad Road**")
                else:
                    st.success("✅ **Good Road**")

    with tabs[1]:
        st.markdown("### 📹 Live Webcam Feed")
        st.write("Start the webcam feed to analyze road conditions in real-time.")

        try:
            webrtc_streamer(
                key="road-detection",
                video_processor_factory=lambda: RoadDetectionTransformer(model),
                media_stream_constraints={"video": True, "audio": False},
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    **About This App:**  
    This application uses a deep learning model to classify roads as either "Good Road" or "Bad Road."  
    Developed with ❤️ using Streamlit and TensorFlow.  
    """)
    st.markdown(
        "🔗 [GitHub Repository](https://github.com/your-repo) | 📧 [Contact Developer](mailto:sahalts777@example.com)")


# Run the App
if __name__ == "__main__":
    main()
