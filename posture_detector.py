import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from transitions import Machine
from typing import Callable


class Postural:
    def on_enter_face_down(self) -> None:
        pass


class PostureDetector:
    def __init__(
        self,
        postural: Postural,
        on_detect: Callable[[vision.FaceDetectorResult, mp.Image, int], None] = None,
    ):
        model = "./blaze_face_short_range.tflite"
        min_detection_confidence = 0.5
        min_suppression_threshold = 0.5

        self.on_detect = on_detect

        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold,
            result_callback=self.save_result,
        )
        detector = vision.FaceDetector.create_from_options(options)
        self.detector = detector

        Machine(
            model=postural,
            states=["unknown", "face_down", "face_up"],
            transitions=[
                ["down", ["face_up"], "face_down"],
                ["up", ["unknown", "face_down"], "face_up"],
            ],
            initial="unknown",
            ignore_invalid_triggers=True,
        )

        self.postural = postural

    def save_result(
        self,
        result: vision.FaceDetectorResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        if self.on_detect:
            self.on_detect(result, unused_output_image, timestamp_ms)

        next_state = "down" if len(result.detections) == 0 else "up"
        self.postural.trigger(next_state)

    def detect_async(self, image: cv2.typing.MatLike, interval: int) -> None:
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face detection using the model.
        self.detector.detect_async(mp_image, interval)

    def close(self) -> None:
        self.detector.close()
