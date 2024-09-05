import cv2
import mediapipe as mp
import pyautogui

hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

def control_mouse():
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hand_detector.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                index_x = int(hand_landmarks.landmark[8].x * screen_width)
                index_y = int(hand_landmarks.landmark[8].y * screen_height)
                thumb_x = int(hand_landmarks.landmark[4].x * screen_width)
                thumb_y = int(hand_landmarks.landmark[4].y * screen_height)
                
                if abs(index_x - thumb_x) < 20 and abs(index_y - thumb_y) < 20:
                    pyautogui.click()
                else:
                    pyautogui.moveTo(index_x, index_y)
        
        cv2.imshow('Hand Gestures', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    control_mouse()