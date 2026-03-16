import cv2

cap = cv2.VideoCapture("http://10.3.10.87:4747/video")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not connected")
        break

    cv2.imshow("DroidCam Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()