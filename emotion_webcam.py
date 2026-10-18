import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print("Press 'c' to capture | 'q' to quit")

captured_emotion = ""

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if captured_emotion:
        cv2.putText(frame, captured_emotion, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]

            captured_emotion = f"{emotion} ({confidence:.1f}%)"
            print("Captured Emotion:", captured_emotion)

        except Exception as e:
            print("Error:", e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()