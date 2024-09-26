import cv2
import time
from utils import visualize
from posture_detector import PostureDetector
import mediapipe as mp
from mediapipe.tasks.python import vision
import sys
from notifier import SlackNotifier
import os
from slack_sdk import WebClient


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


def on_detect(
    result: vision.FaceDetectorResult, unused_output_image: mp.Image, timestamp_ms: int
) -> None:
    global FPS, COUNTER, START_TIME, DETECTION_RESULT

    fps_avg_frame_count = 10

    # Calculate the FPS
    if COUNTER % fps_avg_frame_count == 0:
        FPS = fps_avg_frame_count / (time.time() - START_TIME)
        START_TIME = time.time()

    DETECTION_RESULT = result
    COUNTER += 1


def run() -> None:
    width = 1280
    height = 720

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1

    slack_token = os.environ.get("SLACK_BOT_TOKEN", None)
    client = WebClient(token=slack_token)

    model = SlackNotifier(client, "CHANNEL_ID")

    detector = PostureDetector(
        postural=model,
        on_detect=on_detect,
    )

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        # Capture frame-by-frame
        ret, image = cap.read()
        if not ret:
            sys.exit("Failed to capture image")

        detector.detect_async(image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = "FPS = {:.1f}".format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(
            current_frame,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_DUPLEX,
            font_size,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        if DETECTION_RESULT:
            current_frame = visualize(current_frame, DETECTION_RESULT)

        # Display the resulting frame
        cv2.imshow("frame", current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
