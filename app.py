import streamlit as st
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained model
model = keras.models.load_model("digit_predictor_model.h5")


def predict_digit(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
     # # Convert image to numpy array
    img_array = np.array(image)
    if(img_array.max()>1):
        print("NORMALISING IMAGE: ")
        # pehle se normalised nahi hai
        img_array = img_array/255.0

    print(img_array.shape)
    img_array.reshape(28,28,4)
    img_array = img_array[:,:,0]

    # Reshape the image for prediction
    X_test = img_array.reshape(-1, 28, 28, 1)
    # Perform prediction
    prediction = model.predict(X_test)
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    return predicted_digit


def main():
    st.title("Digit Predictor")

    # Add a file uploader to let users browse their PC and select an image
    uploaded_file = st.file_uploader("Choose an image...", type=["png"])
    # uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # Display the selected image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Predict the digit and display the result
        predicted_digit = predict_digit(image)
        st.write(f"Predicted Digit: {predicted_digit}")


if __name__ == "__main__":
    main()