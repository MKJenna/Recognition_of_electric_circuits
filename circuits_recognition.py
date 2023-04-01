import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import pandas as pd

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
