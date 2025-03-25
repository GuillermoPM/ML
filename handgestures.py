import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pygetwindow as gw

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize the second webcam
cap = cv2.VideoCapture(1)

class HandGestureRecognizer:
    def __init__(self):
        self.gestures = []

    def add_gesture(self, name, condition):
        self.gestures.append((name, condition))

    def recognize_gesture(self, hand_landmarks):
        for name, condition in self.gestures:
            if condition(hand_landmarks):
                return name
        return "Unknown Gesture"

class WindowMover:
    def __init__(self):
        self.is_grabbing = False
        self.start_pos = None
        self.start_hand_pos = None

    def start_grab(self, x, y, hand_x, hand_y):
        self.is_grabbing = True
        self.start_pos = (x, y)
        self.start_hand_pos = (hand_x, hand_y)

    def move_window(self, hand_x, hand_y):
        if self.is_grabbing and self.start_pos and self.start_hand_pos:
            dx = hand_x - self.start_hand_pos[0]
            dy = hand_y - self.start_hand_pos[1]

            # Introduce a threshold to ignore small movements
            threshold = 15
            if abs(dx) > threshold or abs(dy) > threshold:
                window = gw.getActiveWindow()
                if window:
                    window.moveTo(window.left + dx, window.top + dy)
                self.start_hand_pos = (hand_x, hand_y)

    def stop_grab(self):
        self.is_grabbing = False
        self.start_pos = None
        self.start_hand_pos = None



window_mover = WindowMover()

def grab_condition_with_action(hand_landmarks):
    if grab_condition(hand_landmarks):
        # Get the normalized hand position
        hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        screen_width, screen_height = pyautogui.size()
        hand_x = int(hand_x * screen_width)
        hand_y = int(hand_y * screen_height)

        if not window_mover.is_grabbing:
            x, y = pyautogui.position()
            window_mover.start_grab(x, y, hand_x, hand_y)
        else:
            window_mover.move_window(hand_x, hand_y)
        return True
    else:
        if window_mover.is_grabbing:
            window_mover.stop_grab()
        return False

gesture_recognizer = HandGestureRecognizer()

gesture_recognizer.add_gesture("Grab", grab_condition_with_action)
# Initialize the gesture recognizer

def grab_condition(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    

    return (abs(thumb_tip.y  - index_tip.y) < 0.07 and 
            abs(thumb_tip.y - middle_tip.y) < 0.07 and 
            abs(index_tip.y - middle_tip.y) < 0.07)


gesture_recognizer.add_gesture("Grab", grab_condition)

def open_hand_condition(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    return (thumb_tip.y < wrist.y and
            index_tip.y < wrist.y and
            middle_tip.y < wrist.y and
            ring_tip.y < wrist.y and
            pinky_tip.y < wrist.y and
            thumb_tip.y < thumb_ip.y and
            index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y < ring_mcp.y and
            pinky_tip.y < pinky_mcp.y)

gesture_recognizer.add_gesture("Open Hand", open_hand_condition)

def closed_hand_condition(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    return (thumb_tip.y > wrist.y and
            index_tip.y > wrist.y and
            middle_tip.y > wrist.y and
            ring_tip.y > wrist.y and
            pinky_tip.y > wrist.y)

gesture_recognizer.add_gesture("Closed Hand", closed_hand_condition)

def thumbs_up_condition(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    return (thumb_tip.y < thumb_ip.y and
            index_tip.y > index_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y)

gesture_recognizer.add_gesture("Thumbs Up", thumbs_up_condition)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Recognize gesture
            gesture = gesture_recognizer.recognize_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Resize the frame to make it larger
    frame = cv2.resize(frame, (1280, 720))

    # Display the frame
    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()