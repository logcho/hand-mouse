import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

anchor_x, anchor_y = None, None
move_active = False
last_pinch_time = 0

def is_fist(hand_landmarks):
    fingers = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    folded = 0
    for tip in fingers:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        if tip_y > pip_y:
            folded += 1
    return folded == 4

def is_pinch(hand_landmarks, threshold=0.05):
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
    return dist < threshold

while True:
    success, img = cap.read()   
    if not success: 
        continue
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
        curr_x, curr_y = int(wrist.x * 640), int(wrist.y * 480)

        if is_fist(handLms):
            if not move_active:
                anchor_x, anchor_y = curr_x, curr_y
                move_active = True
            else:
                dx = curr_x - anchor_x
                dy = curr_y - anchor_y

                sensitivity = 0.25
                max_speed = 40

                move_x = int(max(-max_speed, min(max_speed, dx * sensitivity)))
                move_y = int(max(-max_speed, min(max_speed, dy * sensitivity)))

                pyautogui.moveRel(move_x, move_y)
        else:
            move_active = False

        if is_pinch(handLms):
            now = time.time()
            if now - last_pinch_time > 1:
                pyautogui.click()
                last_pinch_time = now

    # cv2.imshow("Hand Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()