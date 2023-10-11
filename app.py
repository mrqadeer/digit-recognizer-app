import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import numpy as np

def main():
    st.set_page_config("Model")
    st.title("Digit Classifier App")
    data = ['Sir Irfan Malik', 'Hope To Skill',
            'Artificial Intelligence Course','Mr.Qadeer']
    for name in data:
        st.subheader(name, divider='rainbow')

    # Function to handle URL input and return PIL Image
    def url_handle(url):
        try:
            response = requests.get(url,stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
            return image
        except Exception as e:
            st.error(f"Error occure {e}")
            st.stop()

    # Function to preprocess the image and make predictions
    def prediction(image):
        # Resize image to 28x28
        img_width, img_height = 28, 28
        image = image.resize((img_width, img_height))

        # Convert PIL Image to grayscale and then to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to be between 0 and 1
        image_array = image_array.astype('float32') / 255.0
        
        # Expand dimensions to match the input shape expected by the model
        image_array = np.expand_dims(image_array, axis=-1)
        
        # Load the model if not already loaded
        if 'model' not in st.session_state:
            st.session_state.model = tf.keras.models.load_model('final_model.h5')

            
        # Make predictions
        prediction = st.session_state.model.predict(np.array([image_array]))
        classes = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        
        # Get the predicted class index and label
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = classes[predicted_class_index]
        confidence_level=prediction[0][predicted_class_index]
        with st.expander("Model Predictions",expanded=False):
            st.success(f"Predicted: {predicted_class_index} ({predicted_label})")
            st.success(f"Confidence level: {confidence_level:.2%}")

    # Load class labels

    with st.sidebar:
        with st.expander("What and how this application works",expanded=False):
            st.success("This Application predicts MNIST Digit images You can give it unseen image to be classified.")
            st.success("This Application can accept Local image of format (jpg,jpeg,png) or URL of image. Image should have size \
                       28 by 28 and background black for best prediction.")
        selected = st.selectbox("Select Image source", ( "Local","URL"))

        if selected == 'URL':
            # image_file = st.text_input("Enter Image URL", 'https://www.oreilly.com/api/v2/epubs/9781785880360/files/assets/7a637940-df0c-4df1-9005-f3826536daf8.jpg')
            image_file = st.text_input("Enter Image URL")
            if not image_file:
                st.warning("Please enter your image URL.")
                st.stop()

            try:
                ext=image_file.split('.') # here
                ext_list=['jpg','png','jpeg']
                ex=[e for e in ext_list if e in ext]
                if len(ex)==0:
                    st.error(f"Error Occured your URL does not ends with {ext_list}")
                    st.stop()
                if ex[0] in ext_list:
                    pil_image = url_handle(image_file)
                else:
                    st.warning(f"Your image URL must ends with one of following image extensions {ext_list}")
            except Exception as e:
                st.error(f"Error Occured {e}")
                st.stop()
        elif selected == 'Local':
            image_file = st.file_uploader("Upload your image", type=['jpg', 'jpeg', 'png'])
            if image_file is not None:
                pil_image = Image.open(image_file).convert('L')  # Convert to grayscale
            else:
                st.warning("Please upload an image.")
                st.stop()
        else:
            st.error("No file")
            st.stop()
    st.image(pil_image, 'Image',use_column_width=True)
    if st.button("Predict Image"):
        with st.spinner("Model is Predicting Image"):
            prediction(pil_image)
if __name__=='__main__':
    main()
