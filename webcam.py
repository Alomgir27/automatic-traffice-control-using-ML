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



def update_lane_priority(current_lane, total_vehicle_count):
    '''Update the lane priority based on the current lane and total vehicle count.
    Args:
        lanes (dict): A dictionary of lanes with their vehicle count and last green time.
        current_lane (str): The current lane.
        total_vehicle_count (int): The total vehicle count of the current lane.
    Returns:
        None
    Action:
        The function updates the lane priority based on the current lane and total vehicle count. The weight of the current lane will be updated based on the total vehicle count.
    '''
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
    '''Adjust the traffic lights based on the current lane, vehicle count, and time.
    Args:
        lanes (dict): A dictionary of lanes with their vehicle count and last green time.
        current_lane (str): The current lane.
        total_vehicle_count (int): The total vehicle count of the current lane.
        current_time (float): The current time in seconds.
    Returns:
        None But at a time a lane will open and other will close.
    Action:
        The function controls the traffic lights based on the current lane, vehicle count, and time. A lane will open base on number of vehicle, it's priority and time. also if a lane start it
        will open untill minimum waiting time is over. Also need to care about long starvation of a low weight lane.Also need to care about high traffic of a high weight lane. Have to make it realtic traffic
        control system. We have a raspberry pi. We need to control it. We have to use GPIO pins. We need to use it.
        max weight priority is high.
    '''
     
    #need to make it realtic traffic control system.
    #We have a raspberry pi. We need to control it. We have to use GPIO pins. We need to use it.
    #max weight priority is high.
    
    #update Lane priority
    update_lane_priority(current_lane, total_vehicle_count)
        
    print(LANE_WITH_WEIGHT)
    # Sort lanes based on weight
    LANE_WITH_WEIGHT.sort(key=lambda x: x['weight'], reverse=True)
    
    #Now need to open this lane. but before open it we need to check if it is already open or not. if it is already open then we need check it how long it is open and it is crossed or not maximum green time or not. if it is crossed maximum green time then we need to close it.
    #Before close a lane it should be crossed minimum green time. also if a opened lane vehicles is low compared to others lane and it is crossed minimum green time then we need to close it.
    #also if a lane start it will open untill minimum waiting time is over. Also need to care about long starvation of a low weight lane.Also need to care about high traffic of a high weight lane.

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



# Define function to run inference on a video
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
        results = model.predict(source=frame, conf=0.1, iou=0.45)
        total_vehicle_count = 0
        total_vehicle_count += len(results[0].boxes)
        print(f'Total vehicle count: {total_vehicle_count}')
        current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        print(f'Time: {current_time:.2f} seconds')
        print(f'Video: {video_file}')
        control_traffic_lanes(lanes, current_lane, total_vehicle_count)
        for lane in LANE_WITH_WEIGHT:
            print(f"{lane['lane']} weight: {lane['weight']}")
        for lane in lanes:
            print(f"{lane} is_green: {lanes[lane]['is_green']} last_green: {lanes[lane]['last_green']}")

        time.sleep(2.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# using threads to run inference on multiple videos
def run_multiple_videos(model, video_files):
    threads = []
    #for video_no 1 is lane 1, 2 is lane 2, 3 is lane 3
    for i, video_file in enumerate(video_files):
        current_lane = LANE_ORDER[i]
        thread = threading.Thread(target=run_video, args=(model, video_file, current_lane))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


# Run inference on multiple videos
run_multiple_videos(model, video_files)





