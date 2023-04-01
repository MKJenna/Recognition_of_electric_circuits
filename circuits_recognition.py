import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import pandas as pd
import easyocr
import os
from scipy.spatial.distance import euclidean

import warnings

warnings.filterwarnings('ignore')



def detect_elements(image_path, p=0.5):
    '''
    Recognizes elements in an electrical circuit
        
    Parameters
    ----------
    
    image_path: str
        Location of the image for recognition (example: '/content/C116_D2_P1.jpg')
        
    p: int, default=0.5
        Threshold of model confidence in the prediction, 0 <= p <= 1.
        If an item is recognized with a probability lower
        than the specified one, it will not be included.
        
    
    Returns
    ----------
    df_results : pd.DataFrame
    '''
    # load model with pre-traind weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolo_weights.pt', _verbose=False) 
    # get classes and coordinations
    df_results = model(image_path)
    df_results = df_results.pandas().xyxy[0] 
    df_results = df_results[df_results['confidence'] >= p]
    return df_results




def draw_boxes(image_path, df_results, show_elements=['all'], display_names=False, download=False, download_name='detected_elements.jpg'):
    '''
    Draws frames around found objects 
        
    Parameters
    ----------
    
    image_path: str
        Location of the image for recognition (example: '/content/C116_D2_P1.jpg')
    
    df_results: pd.DataFrame
        The result of object detection with detect_elements
    
    show_elements: list
        What elements should be displayed in the image.
        Use ['all'] if you want to get all elements
        
    
    display_names: bool
        Specifies whether to write its name next to the element
    
    download: bool
        Specifies whether to save image with boxes and texts
    
    download_name: str
        Path for saving image   
    '''
    # load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # select objects for display
    if show_elements == ['all']:
        df_objects = df_results.copy()
    else:
        df_objects = pd.DataFrame(None)
        for element_name in show_elements:
            df_objects = pd.concat([df_objects, df_results[df_results['name'] == element_name]])
    
    # drawing 
    for index, row in df_objects.iterrows():
        cv2.rectangle(image,
                      (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)),
                      (255, 0, 0), 3)
        if display_names == True:
            cv2.putText(image, row['name'], (int(row.xmin)-1, int(row.ymin)-1), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 2, cv2.LINE_AA)
    
    try:
        cv2.imshow('image', image)
    except:
        try:
            cv2_imshow(image)
        except:
            print('Error: cant`t display image.')

    if download == True:
        cv2.imwrite(download_name, image)

        
def find_center(top_left, bottom_right):
    # function for findint center of box
    x_center = (top_left[0] + bottom_right[0]) / 2
    y_center = (top_left[1] + bottom_right[1]) / 2
    return (x_center, y_center)
        
def text_recognition(image_path, df_results, languages=['en'], missing_attribute=None):
    '''
    Recognizing text in a circuit  
        
    Parameters
    ----------
    
    image_path: str
        Location of the image for recognition (example: '/content/C116_D2_P1.jpg')
    
    df_results: pd.DataFrame
        The result of object detection with detect_elements
    
    languages: list
        The list of languages you want to read.
        See details: https://pypi.org/project/easyocr/#:~:text=Note%201%3A%20%5B%27ch_sim%27%2C%27en%27%5D%20is%20the%20list%20of%20languages%20you%20want%20to%20read.%20You%20can%20pass%20several%20languages%20at%20once%20but%20not%20all%20languages%20can%20be%20used%20together.%20English%20is%20compatible%20with%20every%20language%20and%20languages%20that%20share%20common%20characters%20are%20usually%20compatible%20with%20each%20other.
    
    missing_attribute: None or str
        Filler for objects that do not have a text attribute attached
 
    Returns
    ----------
    df_results : pd.DataFrame
        df_results with text atttibutes
    '''
    
    # load model for text recognition
    reader = easyocr.Reader(languages)
    
    # add center points for df_results
    df_results_centers = df_results.copy()
    center = []
    for index, row in df_results_centers.iterrows():
        center.append(find_center([row.xmin, row.ymin], [row.xmax, row.ymax]))
    df_results_centers['xcenter'] = [c[0] for c in center]
    df_results_centers['ycenter'] = [c[1] for c in center]
    
    # divide into textual and non-textual elements 
    not_text = df_results_centers[df_results_centers['name'] != 'text']
    df_text = df_results_centers[df_results_centers['name'] == 'text']
    
    # connect text with object
    text_object = {}
    for text_index, text_row in df_text.iterrows():
        close_object = None
        distance = 1_000_000_000
        for obj_index, obj_row in not_text.iterrows():
            e = euclidean([text_row.xcenter, text_row.ycenter], [obj_row.xcenter, obj_row.ycenter])
            if e < distance:
                distance = e
                close_object = obj_index
        text_object[text_index] = close_object
        
    object_text = {v:i for i, v in text_object.items()}
    
    # add attributes
    attribute_list = []
    for index, row in df_results.iterrows():
        if index not in object_text:
            attribute_list.append(missing_attribute)
        else:
            text_row = df_results.iloc[object_text[index]]
            image = cv2.imread(image_path)
            image_crop = image[int(text_row.ymin)+1:int(text_row.ymax)+1, int(text_row.xmin)-1:int(text_row.xmax-1)].copy()
            text_result = reader.readtext(image_crop, detail=0)
            attribute_list.append(text_result[0])
    df_results['attribute'] = attribute_list
            
    return df_results
