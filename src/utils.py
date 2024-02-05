import pandas as pd

def load_detections(filepath):
    column_labels = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    return pd.read_csv(filepath, header=None, names=column_labels)

def fetch_detections_by_frame(dataframe, frame_id):
    selected_detections = dataframe[dataframe['frame'] == frame_id]
    detection_list = [{
        'frame': det['frame'],
        'bbox': (det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']),
        'conf': det['conf']
    } for _, det in selected_detections.iterrows()]
    return detection_list
