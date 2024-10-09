import cv2
import supervision as sv
from ultralytics import YOLOv10
import torch
import os
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import threading
import requests  # Used to send HTTP requests to NodeMCU

#rasbery pi
# import RPi.GPIO as GPIO



# Load the model
model_path = f'{os.getcwd()}/yolov10/yolov10x.pt'
model = YOLOv10(model_path)

# Define video source (can be a folder or a live camera stream)
video_folder_path = './video'  # Adjust this to your folder path
video_files = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith('.mp4')]

# NodeMCU IP address and control URLs
node_mcu_ip = "http://192.168.0.107"  # Replace with NodeMCU IP
control_urls = {
    "Lane1": f"{node_mcu_ip}/lane1",
    "Lane2": f"{node_mcu_ip}/lane2",
    "Lane3": f"{node_mcu_ip}/lane3"
}


# Initialize lanes with vehicle count and last green time
lanes = {}
for i in range(len(video_files)):
    lanes[f"Lane{i+1}"] = {'last_green': time.time(), 'is_green': False}

#default open first lane
lanes[f"Lane1"] = { 'last_green': time.time(), 'is_green': True}

# Define constants
MAX_GREEN_TIME = 10  # Maximum green time in seconds
MAX_YELLOW_TIME = 5  # Maximum yellow time in seconds
MIN_RED_TIME = 5  # Minimum red time in seconds
MIN_GREEN_TIME = 5  # Minimum green time in seconds
MIN_YELLOW_TIME = 3  # Minimum yellow time in seconds
MAX_RED_TIME = 10  # Maximum red time in seconds
STARVATION_TIME = 50  # Starvation time in seconds
LANE_COUNT = len(video_files)  # Number of lanes
LANE_ORDER = [f"Lane{i+1}" for i in range(LANE_COUNT)]  # Order of lanes
LANE_WITH_WEIGHT = []
for i in range(LANE_COUNT):
    LANE_WITH_WEIGHT.append({'lane': f"Lane{i+1}", 'weight': 1})  # Weight of lanes for traffic control weights priority high to low


# Define a set for emergency lanes
emergency_lanes = set()

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


def close_all_lanes(lanes):
    """Close all lanes."""
    for lane in lanes:
        if lanes[lane]['is_green']:
            lanes[lane]['is_green'] = False

    return

def emergency_open_lane(current_lane):
    """Open the emergency lane."""
    # Already open lane is emergency lane then return and if already open is not emergency lane then close it and open emergency lane
    for lane in lanes:
        if lanes[lane]['is_green']:
            if lane in emergency_lanes:
                return
            else:
                lanes[lane]['is_green'] = False
                lanes[lane]['last_green'] = time.time()
                lanes[current_lane]['is_green'] = True
                lanes[current_lane]['last_green'] = time.time()
                send_command_to_nodemcu(current_lane)
                return


def update_lane_priority(current_lane, total_vehicle_count):
    # Update the weight of the current lane based on the total vehicle count
    for lane in LANE_WITH_WEIGHT:
        if lane['lane'] == current_lane:
            lane['weight'] = total_vehicle_count
            break

    return



def check_starvation(lanes):
    for lane in lanes:
        if time.time() - lanes[lane]['last_green'] > STARVATION_TIME:
            #Close all others lane and open it immediately
            for other_lane in lanes:
                if other_lane != lane:
                    lanes[other_lane]['last_green'] = lanes[other_lane]['last_green'] if lanes[other_lane]['last_green'] == False else time.time()
                    lanes[other_lane]['is_green'] = False
            lanes[lane]['is_green'] = True
            lanes[lane]['last_green'] = time.time()
            send_command_to_nodemcu(lane)
            return True
    return False


# Define function to control traffic lights
def control_traffic_lights(lanes, current_lane, total_vehicle_count):
    update_lane_priority(current_lane, total_vehicle_count)
        
    print(LANE_WITH_WEIGHT)
    # Sort lanes based on weight
    LANE_WITH_WEIGHT.sort(key=lambda x: x['weight'], reverse=True)
    
    # Open the lane with the highest weight
    lane_to_open = LANE_WITH_WEIGHT[0]['lane']

    # Find which lane is opened currently
    current_opened_lane = next((lane for lane in lanes if lanes[lane]['is_green']), None)

    # If no lane is opened currently
    if current_opened_lane is None:
        # Open the lane with the highest weight
        lanes[lane_to_open]['is_green'] = True
        lanes[lane_to_open]['last_green'] = time.time()
        print(f"Opened {lane_to_open}")
        send_command_to_nodemcu(lane_to_open)
        return
    
    
    if check_starvation(lanes):
        return
    # Check if the current opened lane is the same as the lane to open
    if current_opened_lane == lane_to_open:
        # The current opened lane is the same as the lane to open
        # check if it is crossed the maximum green time
        if time.time() - lanes[current_opened_lane]['last_green'] > MAX_GREEN_TIME:
            # Close the current opened lane
            lanes[current_opened_lane]['is_green'] = False
            lanes[current_opened_lane]['last_green'] = time.time()
            #open a new higher priority lane except the current opened lane
            for lane in LANE_WITH_WEIGHT:
                if lane['lane'] != current_opened_lane:
                    lanes[lane['lane']]['is_green'] = True
                    lanes[lane['lane']]['last_green'] = time.time()
                    print(f"Opened {lane['lane']}")
                    send_command_to_nodemcu(lane['lane'])
                    break
            
        else:
            # check if it is crossed the minimum green time
            if time.time() - lanes[current_opened_lane]['last_green'] > MIN_GREEN_TIME:
                # Current lane vehicles is low compared to others higher priority lane then close the current opened lane and open a new higher priority lane
                for lane in LANE_WITH_WEIGHT:
                    if lane['lane'] != current_opened_lane:
                        lanes[current_opened_lane]['is_green'] = False
                        lanes[current_opened_lane]['last_green'] = time.time()
                        lanes[lane['lane']]['is_green'] = True
                        lanes[lane['lane']]['last_green'] = time.time()
                        print(f"Opened {lane['lane']}")
                        break
                    else:
                        break
       
    else:
        # The current opened lane is not the same as the lane to open
        # check if it is crossed the minimum green time
        if time.time() - lanes[current_opened_lane]['last_green'] > MAX_GREEN_TIME:
            # Close the current opened lane
            lanes[current_opened_lane]['is_green'] = False
            lanes[current_opened_lane]['last_green'] = time.time()
            #open a new higher priority lane except the current opened lane
            for lane in LANE_WITH_WEIGHT:
                if lane['lane'] != current_opened_lane:
                    lanes[lane['lane']]['is_green'] = True
                    lanes[lane['lane']]['last_green'] = time.time()
                    print(f"Opened {lane['lane']}")
                    send_command_to_nodemcu(lane['lane'])
                    break
        else:
            # The current lane vehicles is low compared to others lane
            # check if it is crossed the minimum green time
            if time.time() - lanes[current_opened_lane]['last_green'] > MIN_GREEN_TIME:
                # Current lane vehicles is low compared to others higher priority lane then close the current opened lane and open a new higher priority lane
                for lane in LANE_WITH_WEIGHT:
                    if lane['lane'] != current_opened_lane:
                        lanes[current_opened_lane]['is_green'] = False
                        lanes[current_opened_lane]['last_green'] = time.time()
                        lanes[lane['lane']]['is_green'] = True
                        lanes[lane['lane']]['last_green'] = time.time()
                        print(f"Opened {lane['lane']}")
                        send_command_to_nodemcu(lane['lane'])
                        break
                    else:
                        break
    return


# Define function to control traffic lanes
def control_traffic_lanes(lanes, current_lane, total_vehicle_count):
    control_thread = threading.Thread(target=control_traffic_lights, args=(lanes, current_lane, total_vehicle_count))
    control_thread.start()
    control_thread.join()


def run_video(model, video_file, current_lane):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run the model on the frame and extract results
        results = model.predict(source=frame, conf=0.1, iou=0.45)
        total_vehicle_count = 0
        total_person_count = 0
        emergency_vehicle_count = 0

        # Process the detection results
        detected_boxes = results[0].boxes  # assuming 'results' returns a list and 'boxes' is inside
        if detected_boxes:
            for box in detected_boxes:
                class_id = int(box.cls[0])  # Get the class ID of the detected object
                class_name = results[0].names[class_id]  # Get the class name using the ID

                print(f"Class: {class_name}")
                if class_name in ['car', 'bus', 'truck', 'motorcycle', 'vehicle']:
                    total_vehicle_count += 1
                elif class_name in ['ambulance', 'fire truck', 'police car', 'fire brigade']:
                    emergency_vehicle_count += 1
                elif class_name == 'person':
                    total_person_count += 1

        print(f"Total vehicles detected: {total_vehicle_count}")
        print(f"Emergency vehicles detected: {emergency_vehicle_count}")
        print(f"Total persons detected: {total_person_count}")

        if emergency_vehicle_count > 0:
            emergency_lanes.add(current_lane)
        elif emergency_vehicle_count == 0 and current_lane in emergency_lanes:
            emergency_lanes.remove(current_lane)

        
        current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        print(f'Time: {current_time:.2f} seconds')
        print(f'Video: {video_file}')
        
        if len(emergency_lanes) > 0:
            print(f"Emergency lanes: {emergency_lanes}")
            emergency_open_lane(current_lane)
            time.sleep(2.5)
            continue
        # Call the function to control traffic based on vehicle count
        control_traffic_lanes(lanes, current_lane, total_vehicle_count + total_person_count / 4)

        # Display lane weight and green light status
        for lane in LANE_WITH_WEIGHT:
            print(f"{lane['lane']} weight: {lane['weight']}")
        for lane in lanes:
            print(f"{lane} is_green: {lanes[lane]['is_green']} last_green: {lanes[lane]['last_green']}")

        # Sleep for some time (simulate delay)
        time.sleep(2.5)

        # Break on 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# using threads to run inference on multiple videos
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





