import cv2
import mediapipe as mp
import pyautogui
import time
import math
import webbrowser
import os

# ---------------- Mediapipe setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# ---------------- State variables ----------------
anchor_x, anchor_y = None, None
move_active = False
last_pinch_time = 0
last_y_sign_time = 0
gesture_state = "neutral"
gesture_frames = 0
volume_mode = False
last_toggle_time = 0
toggle_start_time = 0
toggle_held = False
HOLD_DURATION = 1.0  # seconds


# ---------------- Helper functions ----------------
def is_fist(hand_landmarks):
    fingers = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]
    folded = 0
    for tip in fingers:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        if tip_y > pip_y:
            folded += 1
    return folded == 4

def is_pinch(hand_landmarks, threshold=0.05, fingers=(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP)):
    finger1 = hand_landmarks.landmark[fingers[0]]
    finger2 = hand_landmarks.landmark[fingers[1]]
    dist = math.hypot(finger1.x - finger2.x, finger1.y - finger2.y)
    return dist < threshold

def is_y_sign(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    thumb_extended = thumb_tip < thumb_ip

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    pinky_extended = pinky_tip < pinky_pip

    folded = 0
    for tip in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP]:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        if tip_y > pip_y:
            folded += 1

    return thumb_extended and pinky_extended and folded == 3

def set_volume(percent):
    percent = max(0, min(100, int(percent)))
    os.system(f"osascript -e 'set volume output volume {percent}'")

def volume_from_distance(hand_landmarks, min_dist=0.02, max_dist=0.3):
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
    vol = (dist - min_dist) / (max_dist - min_dist) * 100
    vol = max(0, min(100, vol))
    return vol

# ---------------- Main loop ----------------
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_gesture = "neutral"

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        wrist = handLms.landmark[mp_hands.HandLandmark.WRIST]
        curr_x, curr_y = int(wrist.x * 640), int(wrist.y * 480)

        # Gesture detection
        if is_fist(handLms):
            current_gesture = "fist"
        elif is_pinch(handLms, fingers=(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP)):
            current_gesture = "pinch"
        elif is_y_sign(handLms):
            current_gesture = "y"

        # Check for toggle gesture
        if is_pinch(handLms, fingers=(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)):
            if not toggle_held:
                toggle_start_time = time.time()
                toggle_held = True
            else:
                # Check if held long enough
                if time.time() - toggle_start_time >= HOLD_DURATION:
                    volume_mode = not volume_mode
                    print("🔊 Volume Mode:", "ON" if volume_mode else "OFF")
                    toggle_held = False  # reset
        else:
            toggle_held = False


        # Debounce for gestures
        if current_gesture == gesture_state:
            gesture_frames += 1
        else:
            gesture_state = current_gesture
            gesture_frames = 1

        # ---------------- Actions ----------------
        if gesture_state == "fist" and gesture_frames > 3 and not volume_mode:
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

        if gesture_state == "pinch" and gesture_frames == 3 and not volume_mode:
            now = time.time()
            if now - last_pinch_time < 0.4:
                pyautogui.doubleClick()
                last_pinch_time = 0
            else:
                pyautogui.click()
                last_pinch_time = now

        if gesture_state == "y" and gesture_frames == 3:
            now = time.time()
            if now - last_y_sign_time > 2:
                print("📺 Opening YouTube...")
                webbrowser.open("https://youtube.com")
                last_y_sign_time = now

        # Volume control when volume_mode ON
        if volume_mode:
            vol = volume_from_distance(handLms)
            set_volume(vol)
            print(f"Volume: {int(vol)}%")

    # cv2.imshow("Hand Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
