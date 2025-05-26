import cv2
import mediapipe as mp
import pyautogui
import random
import util
import numpy as np
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pynput.mouse import Button, Controller
mouse = Controller()

# Variables for Drag & Drop
dragging = False  # To check if drag action is happening
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)



def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


def check_finger_positions(landmarks):    #for brightness and volume control
    def is_finger_extended(tip, pip):
        return landmarks[tip][2] < landmarks[pip][2]  # Tip should be above PIP (y-coordinate lower)

    ring_finger_closed = landmarks[16][2] > landmarks[14][2]  # Ring finger tip below its PIP
    pinky_open = is_finger_extended(20, 18)
    middle_open = is_finger_extended(12, 10)

    return ring_finger_closed and pinky_open and middle_open

def get_distances(frame, landmarks):   #for brightness and volume control
    if len(landmarks) < 21:
        return 0
    (x1, y1), (x2, y2) = (landmarks[4][1], landmarks[4][2]), (landmarks[8][1], landmarks[8][2])
    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return hypot(x2 - x1, y2 - y1)

def get_landmarks(frame, processed, draw, mpHands): #for brightness and volume control
    left_landmarks = []
    right_landmarks = []
    
    if processed.multi_hand_landmarks:
        for handlm in processed.multi_hand_landmarks:
            landmarks = []
            for idx, found_landmark in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(found_landmark.x * width), int(found_landmark.y * height)
                landmarks.append((idx, x, y))

            if handlm == processed.multi_hand_landmarks[0]:
                left_landmarks = landmarks
            elif handlm == processed.multi_hand_landmarks[1]:
                right_landmarks = landmarks

            draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    return left_landmarks, right_landmarks


def get_distance_dd(p1, p2): #for drag and drop
    """Calculate the Euclidean distance between two points."""
    return hypot(p2[0] - p1[0], p2[1] - p1[1])

def convert_to_screen_coords(x, y, frame_width, frame_height): #for drag and drop
    """Convert hand coordinates to screen coordinates."""
    screen_x = int((x / frame_width) * screen_width)
    screen_y = int((y / frame_height) * screen_height)
    return screen_x, screen_y


# Store previous wrist position
prev_wrist_y = None  

# Function to calculate distance between two points
def get_distance_scroll(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])



def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50  and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)






def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol, _ = volRange

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            global dragging  # Ensure dragging is accessible inside the loop
            
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)


            if processed.multi_hand_landmarks:    #for drag and drop
                for hand_landmarks in processed.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    h, w, _ = frame.shape
                
                    # Get landmark positions
                    index_finger_tip = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
                    middle_finger_tip = (int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h))
                    pinky_tip_y = hand_landmarks.landmark[20].y * h
                    pinky_pip_y = hand_landmarks.landmark[18].y * h
                
                    # Draw circles on index and middle fingers
                    cv2.circle(frame, index_finger_tip, 7, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, middle_finger_tip, 7, (255, 0, 0), cv2.FILLED)
                
                    # Calculate distance between index and middle finger tips
                    pinch_distance = get_distance_dd(index_finger_tip, middle_finger_tip)
                
                    # Convert index finger position to screen coordinates
                    cursor_x, cursor_y = convert_to_screen_coords(index_finger_tip[0], index_finger_tip[1], w, h)
                
                    # Check if pinky is closed (tip should be below PIP joint)
                    pinky_closed = pinky_tip_y > pinky_pip_y

                    if pinch_distance < 40 and pinky_closed:  # Pinch detected & pinky closed
                        if not dragging:  # Start dragging
                            pyautogui.mouseDown()
                            dragging = True
                        pyautogui.moveTo(cursor_x, cursor_y)  # Move mouse to new position
                    else:
                        if dragging:  # Release drag when fingers separate
                            pyautogui.mouseUp()
                            dragging = False

                    # Extract landmark positions
                    global prev_wrist_y  # Ensure prev_wrist_y is accessible inside the function

                        # Extract landmark positions
                    wrist_y = hand_landmarks.landmark[0].y * h  # Wrist Y position
                    thumb_tip = (int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h))
                    ring_finger_mcp = (int(hand_landmarks.landmark[13].x * w), int(hand_landmarks.landmark[13].y * h))

                    # Draw circles on thumb tip and ring finger mcp
                    cv2.circle(frame, thumb_tip, 7, (0, 0, 255), cv2.FILLED)
                    cv2.circle(frame, ring_finger_mcp, 7, (0, 0, 255), cv2.FILLED)

                    # Check thumb-ring finger distance
                    thumb_ring_distance = get_distance_scroll(thumb_tip, ring_finger_mcp)

                    # Scrolling only if thumb tip is close to ring finger tip
                    if thumb_ring_distance < 50:  # Adjust threshold based on testing
                        if prev_wrist_y is not None:
                            scroll_speed = int((prev_wrist_y - wrist_y) * 10)  # Scale for sensitivity
                            if abs(scroll_speed) > 1:  # Ignore tiny movements
                                pyautogui.scroll(scroll_speed)

                        prev_wrist_y = wrist_y  # Update previous wrist position   

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))


            detect_gesture(frame, landmark_list, processed)

            left_landmarks, right_landmarks = get_landmarks(frame, processed, draw, mpHands)

            if left_landmarks and check_finger_positions(left_landmarks):
                left_distance = get_distances(frame, left_landmarks)
                b_level = np.interp(left_distance, [50, 220], [0, 100])
                sbc.set_brightness(int(b_level))

            if right_landmarks and check_finger_positions(right_landmarks):
                right_distance = get_distances(frame, right_landmarks)
                vol = np.interp(right_distance, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()





if __name__ == '__main__':
    main()




# for mouse movement thumb close  if thumb opened move won't move 
# for left click thumb close index finger bend
# for right click thumb close middle finger bend
# for double click thumb close and index,middle finger bend
# screenshot thumb close and all fingers close

# for brightness control single hand will work ---> but ring finger should be closed then only brightness will work
# for voulme control both hands should be in frame ---> but right hand works for voulme control and left hand works for brightnrss control

#for drag and drop pinky finger should be closed ,Pinch between the index finger tip (8) and middle finger tip (12) 
# drag snd drop[index and middle fingers move cheyali like scissor movement]

#scroll --->Thumb tip near Ring Finger end (landmark 13) → Scrolling Works,Thumb Moves Away/opens → Scrolling Stops
# [wrist landmark(0)]Move Wrist Up → Scrolls Up,Move Wrist Down → Scrolls Down