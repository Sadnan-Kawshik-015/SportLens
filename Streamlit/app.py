import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

MODEL = tf.keras.models.load_model("Model/SportLens.h5")
# CLASS_NAME = ['Air Hockey', 'Ampute Football', 'Archery', 'Arm Wrestling', 'Axe Throwing', 'Balance Beam', 'Barell Racing', 'Baseball', 'Basketball', 'Baton Twirling', 'Bike Polo', 'Billiards', 'BMX', 'Bobsled', 'Bowling', 'Boxing', 'Bull Riding', 'Bungee Jumping', 'Canoe Slamon', 'Cheerleading', 'Chuckwagon Racing', 'Cricket', 'Croquet', 'Curling', 'Disc Golf', 'Fencing', 'Field Hockey', 'Figure Skating Men', 'Figure Skating Pairs', 'Figure Skating Women', 'Fly Fishing', 'Football', 'Formula 1 Racing', 'Frisbee', 'Gaga', 'Giant Slalom', 'Golf', 'Hammer Throw', 'Hang Gliding', 'Harness Racing', 'High Jump', 'Hockey', 'Horse Jumping', 'Horse Racing', 'Horseshoe Pitching', 'Hurdles', 'Hydroplane Racing', 'Ice Climbing', 'Ice Yachting', 'Jai Alai', 'Javelin', 'Jousting', 'Judo', 'Lacrosse', 'Log Rolling', 'Luge', 'Motorcycle Racing', 'Mushing', 'NASCAR Racing', 'Olympic Wrestling', 'Parallel Bar', 'Pole Climbing', 'Pole Dancing', 'Pole Vault', 'Polo', 'Pommel Horse', 'Rings', 'Rock Climbing', 'Roller Derby', 'Rollerblade Racing', 'Rowing', 'Rugby', 'Sailboat Racing', 'Shot Put', 'Shuffleboard', 'Sidecar Racing', 'Ski Jumping', 'Sky Surfing', 'Skydiving', 'Snow Boarding', 'Snowmobile Racing', 'Speed Skating', 'Steer Wrestling', 'Sumo Wrestling', 'Surfing', 'Swimming', 'Table Tennis', 'Tennis', 'Track Bicycle', 'Trapeze', 'Tug Of War', 'Ultimate', 'Uneven Bars', 'Volleyball', 'Water Cycling', 'Water Polo', 'Weightlifting', 'Wheelchair Basketball', 'Wheelchair Racing', 'Wingsuit Flying']

CLASS_NAMES = ['air hockey',
 'ampute football',
 'archery',
 'arm wrestling',
 'axe throwing',
 'balance beam',
 'barell racing',
 'baseball',
 'basketball',
 'baton twirling',
 'bike polo',
 'billiards',
 'bmx',
 'bobsled',
 'bowling',
 'boxing',
 'bull riding',
 'bungee jumping',
 'canoe slamon',
 'cheerleading',
 'chuckwagon racing',
 'cricket',
 'croquet',
 'curling',
 'disc golf',
 'fencing',
 'field hockey',
 'figure skating men',
 'figure skating pairs',
 'figure skating women',
 'fly fishing',
 'football',
 'formula 1 racing',
 'frisbee',
 'gaga',
 'giant slalom',
 'golf',
 'hammer throw',
 'hang gliding',
 'harness racing',
 'high jump',
 'hockey',
 'horse jumping',
 'horse racing',
 'horseshoe pitching',
 'hurdles',
 'hydroplane racing',
 'ice climbing',
 'ice yachting',
 'jai alai',
 'javelin',
 'jousting',
 'judo',
 'lacrosse',
 'log rolling',
 'luge',
 'motorcycle racing',
 'mushing',
 'nascar racing',
 'olympic wrestling',
 'parallel bar',
 'pole climbing',
 'pole dancing',
 'pole vault',
 'polo',
 'pommel horse',
 'rings',
 'rock climbing',
 'roller derby',
 'rollerblade racing',
 'rowing',
 'rugby',
 'sailboat racing',
 'shot put',
 'shuffleboard',
 'sidecar racing',
 'ski jumping',
 'sky surfing',
 'skydiving',
 'snow boarding',
 'snowmobile racing',
 'speed skating',
 'steer wrestling',
 'sumo wrestling',
 'surfing',
 'swimming',
 'table tennis',
 'tennis',
 'track bicycle',
 'trapeze',
 'tug of war',
 'ultimate',
 'uneven bars',
 'volleyball',
 'water cycling',
 'water polo',
 'weightlifting',
 'wheelchair basketball',
 'wheelchair racing',
 'wingsuit flying']


def read_file_as_image(data):

    image = np.array(data)
    image = np.array(Image.fromarray((image).astype(np.uint8)).resize((224, 224)))
    image = image/255.

    return image


def main():
    st.title("SportLens")

    st.header("About")
    st.write("Introducing SportLens, a powerful machine learning model utilizing the renowned VGG16 architecture. "
             "Trained on the Kaggle 100 Sports Dataset, SportLens accurately identifies various sports from images."
             " With a comprehensive collection of 1000 sports categories, it offers precise and efficient sports classification.")
    st.subheader("Model Used: ")
    st.write("VGG16(feature extractor,Flatten Layer,Dense Layer)")
    st.subheader("Dataset")
    st.markdown("[100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)")

    # Header with GitHub link
    st.header("GitHub Repository")
    st.markdown("[SportLens Repository](https://github.com/Sadnan-Kawshik-015/SportLens)")

    # Upload image through Streamlit widget
    uploaded_image = st.file_uploader("Upload an image of a sport", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        pil_image = Image.open(uploaded_image)
        image = read_file_as_image(pil_image)
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        # print(type(predictions[0][0]))
        sorted_predictions = np.sort(predictions[0])

        # for element in sorted_predictions[-5:]:
        #  # Convert each element to percentage form
        #  percentage_value = element * 100.0
        #
        #  # Display the element in a Streamlit card-like format
        #  st.write(
        #   f"Original: {element:.4f} - Percentage: {percentage_value:.2f}%"
        #  )

        # Process the image (you can replace this with your own image processing logic)
        processed_text = predicted_class.title()

        # Display the processed text
        st.subheader("The Given Image is of: "+processed_text)
        st.subheader("Confidence: "+"{:.4f}".format(confidence*100)+"%")
        # Footer with your name
    st.markdown("---")
    st.markdown("Created by: Sadnan Kibria Kawshik")
    st.markdown("[GitHub](https://github.com/Sadnan-Kawshik-015)")


def process_image(image):
    # Placeholder function for image processing
    # Replace this with your own image processing logic
    # For example, you can use OCR (Optical Character Recognition) to extract text from the image
    return "This is a placeholder for processed text."

if __name__ == "__main__":
    main()
