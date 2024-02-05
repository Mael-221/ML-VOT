from scipy.optimize import linear_sum_assignment
from utils import load_detections, fetch_detections_by_frame
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import cv2


def calculate_iou(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB

    # Determine the coordinates of the intersection rectangle
    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax1 + aw, bx1 + bw)
    inter_bottom = min(ay1 + ah, by1 + bh)

    # Compute the area of intersection rectangle
    inter_width = max(inter_right - inter_left, 0)
    inter_height = max(inter_bottom - inter_top, 0)
    intersection_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    areaA = aw * ah
    areaB = bw * bh

    # Compute the area of the union
    union_area = areaA + areaB - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def match_detections_to_tracks(tracks, detections, min_iou=0.5):
    matched_pairs = []
    remaining_tracks = list(range(len(tracks)))
    remaining_detections = list(range(len(detections)))

    # Early exit if no tracks or detections
    if not tracks or not detections:
        return matched_pairs, remaining_detections, remaining_tracks

    # Compute IoU matrix between all detections and tracks
    iou_scores = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for i, detection in enumerate(detections):
        for j, track in enumerate(tracks):
            iou_scores[i, j] = calculate_iou(detection['bbox'], track['bbox'])

    # Hungarian algorithm to find the best matches
    rows, cols = linear_sum_assignment(-iou_scores)
    for i, j in zip(rows, cols):
        if iou_scores[i, j] >= min_iou:
            matched_pairs.append((i, j))
            remaining_detections.remove(i)
            remaining_tracks.remove(j)

    return matched_pairs, remaining_detections, remaining_tracks

def tracks_update(matched_pairs, new_detections, detected_objects, existing_tracks, current_frame):
    # Update existing tracks with matched detections
    for detection_idx, track_idx in matched_pairs:
        existing_tracks[track_idx]['bbox'] = detected_objects[detection_idx]['bbox']
        existing_tracks[track_idx]['history'].append({
            'frame': current_frame,
            'bbox': detected_objects[detection_idx]['bbox']
        })

    # Create new tracks for detections that weren't matched to any track
    for detection_idx in new_detections:
        new_track = {
            'id': len(existing_tracks) + 1, 
            'bbox': detected_objects[detection_idx]['bbox'],
            'history': [{
                'frame': current_frame,
                'bbox': detected_objects[detection_idx]['bbox']
            }]
        }
        existing_tracks.append(new_track)

    
def save_tracking(tracks, output_file):
    with open(output_file, 'w') as f:
        for track in tracks:
            for det in track['history']:
                f.write(f"{det['frame']},{track['id']},{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]},1,-1,-1,-1\n")


def process_tracking(det_path, img_folder_path, output_csv, nb_frame=525, threshold=0.5):
    # Load detection data
    detections_df = load_detections(det_path)
    
    # Initialize tracks
    tracks = []
    
    # Construct the list of image files
    img_files = sorted(glob(os.path.join(img_folder_path, "*.jpg")))[:nb_frame]
    
    # Process each image file
    for img_file in tqdm(img_files):
        frame_number = int(os.path.basename(img_file).split('.')[0])
        current_detections = fetch_detections_by_frame(detections_df, frame_number)
        matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(tracks, current_detections, threshold)
        tracks_update(matches, unmatched_detections, current_detections, tracks, frame_number)
    
    # Save the tracking results
    save_tracking(tracks, output_csv)
    return tracks



def draw_tracking_results(image_dir, tracks):
    """
    Draw bounding boxes and IDs on images to visualize tracking results.
    :param image_dir: Directory containing images
    :param tracks: List of tracks, each with an ID and a history of detections
    """
    image_files = sorted(os.listdir(image_dir))
    
    for frame, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        
        # Iterate through each track and its detections
        for track in tracks:
            for detection in track['history']:
                if detection['frame'] == frame:
                    # Extract the bounding box and draw it along with the track ID
                    bbox = detection['bbox']
                    bb_left, bb_top, bb_width, bb_height = [int(coord) for coord in bbox]  # Ensure coordinates are integers
                    
                    # Calculate bottom-right corner of the bounding box
                    bb_right = bb_left + bb_width
                    bb_bottom = bb_top + bb_height
                    
                    # Draw bounding box
                    cv2.rectangle(image, (bb_left, bb_top), (bb_right, bb_bottom), (0, 255, 0), 2)
                    
                    # Draw ID
                    cv2.putText(image, str(track['id']), (bb_left, bb_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow('Tracking Results', image)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    
    cv2.destroyAllWindows()

def main():
    img_folder_path = 'ADL-Rundle-6/img1'  # Path to the image folder
    detection_file = 'ADL-Rundle-6/det/det.txt'  # Path to the detection file
    output_csv = 'ADL-Rundle-6.txt'  # Output CSV file name for the tracking results

    # Call the updated process_tracking function
    tracks = process_tracking(detection_file, img_folder_path, output_csv)
    draw_tracking_results(img_folder_path, tracks)

if __name__ == "__main__":
    main()
