import streamlit as st
import numpy as np
from PIL import Image
import boto3
import io
import json
import pandas as pd
from datetime import datetime
import os



def main():
    st.title("Multimodal Model Predictions")

    # Get tabular data from user input
    class_options = [
        "flower", "mountain", "cake", "boat", "cat", "dog", "airplane", "person", "llama", "cartoon"
    ]
    image_class = st.selectbox("Select image class:", class_options)
    date = st.text_input("Enter date:", "2024/01/11")
    bar = st.number_input("Enter bar:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    baz = st.radio("baz:", ["Yes", "No"]) == "Yes"
    xgt = st.number_input("Enter xgt:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    qgg = st.number_input("Enter qgg:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    lux = st.number_input("Enter lux:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    wsg = st.number_input("Enter wsg:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    yyz = st.number_input("Enter yyz:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    drt = st.number_input("Enter drt:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    gox = st.number_input("Enter gox:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    foo = st.number_input("Enter foo:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    boz = st.number_input("Enter boz:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    fyt = st.radio("fyt:", ["Yes", "No"]) == "Yes"
    lgh = st.radio("lgh:", ["Yes", "No"]) == "Yes"
    hrt = st.number_input("Enter hrt:", value=0.0, min_value=-1000000.0, max_value=1000000.0)
    juu = st.number_input("Enter juu:", value=0.0, min_value=-1000000.0, max_value=1000000.0)


    # Get image from user input
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        # Process image as needed
        processed_image = preprocess_image(image_file)

        
        tabular_data = np.array([image_class,date, bar, baz, xgt, qgg, lux, wsg, yyz, drt, gox, foo, boz, fyt, lgh, hrt, juu])

        #Column names
        column_names = ['image_class','date', 'bar', 'baz', 'xgt', 'qgg', 'lux', 'wsg', 'yyz', 'drt', 'gox', 'foo', 'boz', 'fyt', 'lgh', 'hrt', 'juu']

        # Reshape the processed image to (1, 224, 224, 3)
        processed_image = processed_image.reshape((1, 224, 224, 3))

        processed_tabular_data = preprocess_tabular_data(column_names, tabular_data)
        print(processed_tabular_data)
        
        # Make prediction
        prediction = get_prediction(processed_tabular_data, processed_image)

        st.write(f"# Prediction: {prediction}")

def preprocess_image(image_file):
    # Load image data
    image_size = (224, 224)

    try:
        # Use Image.open to open the image
        image = Image.open(image_file)

        # Resize and convert to array
        img_resized = np.array(image.resize(image_size))

        return img_resized/255.0
    except Exception as e:
        print(f"Error loading image: {e}")

def preprocess_tabular_data(column_names, tabular_data):
    # Convert to dataframe
    tabular_data_df = pd.DataFrame([tabular_data], columns=column_names)

    # Convert date to datetime
    tabular_data_df["date"] = pd.to_datetime(tabular_data_df["date"])
    
    # Reference date
    reference_date = datetime.strptime('2019-02-27', '%Y-%m-%d')

    # Calculate days since reference date
    tabular_data_df['days_since'] = (tabular_data_df['date'] - reference_date).dt.days
    tabular_data_df = tabular_data_df.drop('wsg', axis=1)
    tabular_data_df = tabular_data_df.drop('date', axis=1)

    # Get boolean columns 
    bool_columns = ['baz', 'fyt', 'lgh']
    tabular_data_df[bool_columns] = tabular_data_df[bool_columns].replace({'True': True, 'False': False})
    tabular_data_df[bool_columns] = tabular_data_df[bool_columns].astype(int)

    # Specify all possible classes
    all_classes = ['airplane', 'boat', 'cake', 'cartoon', 'cat', 'dog', 'flower', 'llama', 'mountain', 'person']

    # Add columns for all possible classes with zeros
    tabular_data_df = pd.concat([tabular_data_df, pd.DataFrame(0, index=tabular_data_df.index, columns=all_classes)], axis=1)

    # Update the relevant column based on the "image_class"
    tabular_data_df.loc[tabular_data_df.index, tabular_data_df['image_class']] = 1

    # Drop the original "image_class" column
    tabular_data_df.drop('image_class', axis=1, inplace=True)

    # Convert columns to float
    columns_to_convert = ["bar", "xgt", "qgg", "lux", "yyz", "drt", "gox", "foo", "boz", "hrt", "juu"]

    # Convert specified columns to float
    tabular_data_df[columns_to_convert] = tabular_data_df[columns_to_convert].astype(float)

    # Normalize with the sames training min and max
    min_max_dict = {
    'bar': {'min': -122.718563, 'max': 1719.200416},
    'xgt': {'min': -659.009264, 'max': 756.848769},
    'qgg': {'min': -26.393565, 'max': 25.452515},
    'lux': {'min': 3577.806382, 'max': 7627.667867},
    'yyz': {'min': -26.589758, 'max': 25.742596},
    'drt': {'min': -263127.335321, 'max': -0.001729},
    'gox': {'min': -26.977454, 'max': 26.936961},
    'foo': {'min': -5.853642, 'max': 66.001361},
    'boz': {'min': 0.023978, 'max': 103.649905},
    'hrt': {'min': -735.738392, 'max': 266.571596},
    'juu': {'min': 0.473126, 'max': 866.895515},
    }

    # Extract the columns to normalize
    columns_to_normalize = ['bar', 'xgt', 'qgg', 'lux', 'yyz', 'drt', 'gox', 'foo', 'boz', 'hrt', 'juu']

    # Normalize the columns
    for col in columns_to_normalize:
        tabular_data_df[col] = (tabular_data_df[col] - min_max_dict[col]['min']) / (min_max_dict[col]['max'] - min_max_dict[col]['min'])

    tabular_data_np_array = tabular_data_df.values

    return tabular_data_np_array

def get_prediction(tabular_data, image_array):
    # Convert tabular_data to a list
    tabular_data_list = tabular_data.tolist()

    #tabular_data_list = [ 0.0591344237615152, 1, 0.5588512085211652, 0.2219173855814699, 0.8597296204618212, 0.3144921888322008, 0.9999874100782428, 0.1361326506474088, 0.4262547610601505, 0.0005181880377691, 1, 0, 0.590296209079183, 0.4241634480512994, 0.967978042086002, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    
    # Create the payload
    print(tabular_data_list)
    payload = {
        "instances": [
            {"input_1": tabular_data_list, "input_2": image_array.tolist()}
        ]
    }

    # Convert payload to JSON-formatted string
    payload_json = json.dumps(payload)

    client = boto3.client('sagemaker-runtime', 'us-east-1', aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])

    # Make the request
    response = client.invoke_endpoint(
        EndpointName="test-endpoint-3", 
        ContentType="application/json",  # Use application/json content type
        Body=payload_json.encode('utf-8')  # Encode the JSON string to bytes
    )

    result = response['Body'].read()

    # Process the result as needed
    # Convert the bytes string to a regular string
    json_string = result.decode('utf-8')

    # Parse the JSON string into a dictionary
    prediction_result = json.loads(json_string)

    # Access the predicted value
    predicted_value = prediction_result["predictions"][0][0]


    return predicted_value

if __name__ == '__main__':
    main()
