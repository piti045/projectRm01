from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyttsx3

engine = pyttsx3.init()
last_word = ""

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)  # ใช้กล้องโน้ตบุ๊ก

def detect_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    thumb = hand_landmarks.landmark[4]
    index_finger = hand_landmarks.landmark[8]
    middle_finger = hand_landmarks.landmark[12]
    pinky = hand_landmarks.landmark[20]

    # HELLO (นิ้วชี้ขึ้น)
    if index_finger.y < wrist.y and middle_finger.y > wrist.y:
        return "HELLO"

    # YES (กำมือ)
    if (thumb.y > wrist.y and index_finger.y > wrist.y and middle_finger.y > wrist.y):
        return "YES"

    # I LOVE YOU (โป้ง + ชี้ + ก้อย)
    if thumb.y < wrist.y and index_finger.y < wrist.y and pinky.y < wrist.y:
        return "I LOVE YOU"

    return "UNKNOWN"


def generate_frames():
    global last_word

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_text = "No hand"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_text = detect_gesture(hand_landmarks)

                # 🔊 พูดออกลำโพง เมื่อเปลี่ยนคำ
                if gesture_text != last_word and gesture_text != "UNKNOWN":
                    engine.say(gesture_text)
                    engine.runAndWait()
                    last_word = gesture_text

        cv2.putText(frame, gesture_text, (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True, use_reloader=False)
