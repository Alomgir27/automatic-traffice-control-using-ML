import cv2
import supervision as sv
from ultralytics import YOLOv10
import torch
import os
import threading
import time
import requests  # Used to send HTTP requests to NodeMCU

# Load the model
model_path = f'{os.getcwd()}/yolov10/yolov10x.pt'
model = YOLOv10(model_path)

# Define video source (can be a folder or a live camera stream)
video_folder_path = './video'  # Adjust this to your folder path
video_files = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith('.mp4')]

# Initialize lanes with vehicle count and last green time
lanes = {}
for i in range(len(video_files)):
    lanes[f"Lane{i+1}"] = {'last_green': time.time(), 'is_green': False}

# Default open first lane
lanes[f"Lane1"] = {'last_green': time.time(), 'is_green': True}

# NodeMCU IP address and control URLs
node_mcu_ip = "http://192.168.0.107"  # Replace with NodeMCU IP
control_urls = {
    "Lane1": f"{node_mcu_ip}/lane1",
    "Lane2": f"{node_mcu_ip}/lane2",
    "Lane3": f"{node_mcu_ip}/lane3"
}

# Define constants
MAX_GREEN_TIME = 5  # Maximum green time in seconds
MIN_GREEN_TIME = 3  # Minimum green time in seconds
STARVATION_TIME = 15  # Starvation time in seconds
LANE_COUNT = len(video_files)  # Number of lanes
LANE_ORDER = [f"Lane{i+1}" for i in range(LANE_COUNT)]
LANE_WITH_WEIGHT = [{'lane': f"Lane{i+1}", 'weight': 1} for i in range(LANE_COUNT)]


def update_lane_priority(current_lane, total_vehicle_count):
    """Update the lane priority based on the vehicle count."""
    for lane in LANE_WITH_WEIGHT:
        if lane['lane'] == current_lane:
            lane['weight'] = total_vehicle_count
            break
    return


def check_starvation(lanes):
    """Check if any lane is starved for too long and needs to be opened."""
    for lane in lanes:
        if time.time() - lanes[lane]['last_green'] > STARVATION_TIME:
            for other_lane in lanes:
                lanes[other_lane]['is_green'] = False
            lanes[lane]['is_green'] = True
            lanes[lane]['last_green'] = time.time()
            return True
    return False


def send_command_to_nodemcu(lane):
    """Send the HTTP request to NodeMCU to open the specific lane."""
    print(f"Sending command to NodeMCU: {control_urls[lane]}")
    try:
        response = requests.get(control_urls[lane])
        if response.status_code == 200:
            print(f"Successfully opened {lane}")
        else:
            print(f"Failed to open {lane}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error communicating with NodeMCU: {e}")


def control_traffic_lights(lanes, current_lane, total_vehicle_count):
    """Control traffic lights based on lane weights and vehicle counts."""
    update_lane_priority(current_lane, total_vehicle_count)
    
    LANE_WITH_WEIGHT.sort(key=lambda x: x['weight'], reverse=True)
    lane_to_open = LANE_WITH_WEIGHT[0]['lane']
    current_opened_lane = next((lane for lane in lanes if lanes[lane]['is_green']), None)

    if current_opened_lane is None or check_starvation(lanes):
        lanes[lane_to_open]['is_green'] = True
        lanes[lane_to_open]['last_green'] = time.time()
        send_command_to_nodemcu(lane_to_open)
        return

    if current_opened_lane == lane_to_open:
        if time.time() - lanes[current_opened_lane]['last_green'] > MAX_GREEN_TIME:
            lanes[current_opened_lane]['is_green'] = False
            lanes[lane_to_open]['is_green'] = True
            lanes[lane_to_open]['last_green'] = time.time()
            send_command_to_nodemcu(lane_to_open)
    else:
        if time.time() - lanes[current_opened_lane]['last_green'] > MIN_GREEN_TIME:
            lanes[current_opened_lane]['is_green'] = False
            lanes[lane_to_open]['is_green'] = True
            lanes[lane_to_open]['last_green'] = time.time()
            send_command_to_nodemcu(lane_to_open)
    return


def run_video(model, video_file, current_lane):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.1, iou=0.45)
        total_vehicle_count = len(results[0].boxes)
        print(f'Total vehicle count in {current_lane}: {total_vehicle_count}')
        
        control_traffic_lights(lanes, current_lane, total_vehicle_count)

        time.sleep(2.5)

    cap.release()


def run_multiple_videos(model, video_files):
    threads = []
    for i, video_file in enumerate(video_files):
        current_lane = LANE_ORDER[i]
        thread = threading.Thread(target=run_video, args=(model, video_file, current_lane))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


# Run inference on multiple videos
run_multiple_videos(model, video_files)
