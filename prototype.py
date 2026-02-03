import cv2
import pyttsx3
from ultralytics import YOLO
import time
import threading
import numpy as np
import datetime

# --- 1. VOICE FUNCTION ---
def speak_now(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0) 
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    thread = threading.Thread(target=run)
    thread.start()

# ==========================================
#      HERO ENTRY
# ==========================================
print("--- SYSTEM BOOTING ---")
speak_now("Powering on Smart Glasses. Let's see the world through the third eye. Now you can use it..")
print("🔊 Audio Playing... Waiting 5 seconds...")
time.sleep(5) 

# ==========================================
#      SYSTEM LOAD
# ==========================================
print("Loading AI...")
# Note: imgsz hata diya hai taaki accuracy badh jaye
model = YOLO('yolov8n.pt') 

print("Starting Camera...")
cap = cv2.VideoCapture(1) # Changed to 0 for default webcam. Change back to 1 if using external.
cap.set(3, 640)
cap.set(4, 480)

# Colors
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)   
GREEN = (0, 255, 0)    
RED = (0, 0, 255)
GREY = (100, 100, 100) # Un-focused objects ke liye
WHITE = (255, 255, 255)

target_objects = ['bottle', 'cell phone', 'person', 'cup', 'laptop', 'mouse', 'keyboard', 'chair']
last_speak_time = 0
last_dark_warning = 0
current_object = ""

print("\n>>> SYSTEM ONLINE: CENTER FOCUS MODE <<<")

while True:
    success, img = cap.read()
    if not success:
        break

    height, width, _ = img.shape
    
    # --- 1. DRAW GRID (Scifi Look) ---
    # Vertical Lines
    for x in range(0, width, 80): 
        cv2.line(img, (x, 0), (x, height), (40, 40, 40), 1)
    # Horizontal Lines
    for y in range(0, height, 80):
        cv2.line(img, (0, y), (width, y), (40, 40, 40), 1)

    # --- 2. DEFINE FOCUS ZONE (Beech ka hissa) ---
    # Hum chahte hain sirf beech ke 250 pixels mein detection ho
    # Screen width 640 hai. Center 320 hai.
    focus_box_x1 = (width // 2) - 130
    focus_box_x2 = (width // 2) + 130
    
    # Draw Brackets (Taaki pata chale kahan dekhna hai)
    cv2.rectangle(img, (focus_box_x1, 50), (focus_box_x2, height-50), WHITE, 1)
    cv2.putText(img, "[ TARGET ZONE ]", (focus_box_x1+20, 40), FONT, 0.5, WHITE, 1)

    # --- 3. DARKNESS CHECK ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 40: 
        cv2.putText(img, "LOW LIGHT", (200, 240), FONT, 0.8, RED, 2)
        if (time.time() - last_dark_warning > 15): 
            speak_now("Low light detected.")
            last_dark_warning = time.time()

    # --- 4. AI DETECTION ---
    results = model(img, stream=True, verbose=False)
    
    danger_close = False 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            
            if conf > 0.55:
                cls = int(box.cls[0])
                obj_name = model.names[cls]

                if obj_name in target_objects:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate Object Center (Object ka beech ka point)
                    obj_center_x = (x1 + x2) // 2
                    
                    # --- CRITICAL LOGIC: KYA OBJECT FOCUS ZONE MEIN HAI? ---
                    is_in_focus = (focus_box_x1 < obj_center_x < focus_box_x2)

                    if is_in_focus:
                        # === ACTIVE ZONE (Speak & Red/Green) ===
                        
                        # DISTANCE LOGIC (Fixed - Ab 400px height chahiye)
                        box_height = y2 - y1
                        if box_height > 400: 
                            danger_close = True
                            color = RED
                            label = f"STOP! {obj_name}"
                        else:
                            color = GREEN
                            label = f"{obj_name}"
                        
                        # Box Banao (Bright Color)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1-10), FONT, 0.8, color, 2)

                        # AUDIO LOGIC
                        current_time = time.time()
                        if danger_close:
                            if (current_time - last_speak_time > 2):
                                speak_now(f"Stop! {obj_name} too close.")
                                last_speak_time = current_time
                        elif (obj_name != current_object) or (current_time - last_speak_time > 3):
                            print(f"🔊 Focused: {obj_name}") 
                            speak_now(f"{obj_name}") # Short me bolega
                            last_speak_time = current_time
                            current_object = obj_name
                            
                    else:
                        # === PASSIVE ZONE (Side Objects) ===
                        # Inko sirf Grey dikhao aur chup raho
                        cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)
                        # No Sound

    # --- HUD INFO ---
    current_time_str = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, f"MODE: FOCUS | BAT: 98% | {current_time_str}", (20, 460), FONT, 0.5, CYAN, 1)

    # Crosshair Center
    cv2.circle(img, (width//2, height//2), 5, CYAN, -1)

    # Red Screen Flash only if Focused Danger
    if danger_close:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img) 

    cv2.imshow('Phoenix Project - Focus View', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()