import torch

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
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolo_weights.pt') 
    # get classes and coordinations
    df_results = model(image_path)
    df_results = df_results.pandas().xyxy[0] 
    df_results = df_results[df_results['confidence'] >= p]
    return df_results
