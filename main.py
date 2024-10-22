import cv2
import os
from dotenv import load_dotenv
import ultralytics
from tracker import Tracker
import time
from ActionsEstLoader import TSSTG
import telegram

load_dotenv()

CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")

TELEGRAM_TOKEN = os.getenv("BOT_TELEGRAM_API_KEY")
bot = telegram.Bot(token=TELEGRAM_TOKEN)

TIME_DELAY = 100

def get_chat_id():
    updates = bot.get_updates()
    for update in updates:
        if update.message:
            chat_id = update.message.chat.id
            print(f"Chat ID: {chat_id}")
            return chat_id
    return None

def main():
    cap = cv2.VideoCapture(f"rtsp://admin:{CAMERA_PASSWORD}@192.168.1.2:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
    tracker = Tracker(weights="weight/yolov8n-pose_ncnn_model")
    action_model = TSSTG()

    last_inference_time = time.time()
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        elapsed_time = current_time - last_inference_time
        if elapsed_time < TIME_DELAY / 1000:
            continue
        # Reduce size of the frame twice
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        pose = tracker.get_pose_track(frame)
        pts = pose.get_points()
        out = action_model.predict(pts, frame.shape[:2])
        action_name = action_model.class_names[out[0].argmax()]
        if action_name == "Fall Down":
            # Send notification by telegram
            print("Fall Down")
            chat_id = get_chat_id()
            bot.sendMessage(chat_id=chat_id, text="Fall Down")

        show_frame = pose.plot()
        cv2.imshow('frame', show_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        last_inference_time = time.time()

        print(f"FPS: {1 / (time.time() - start_time)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()