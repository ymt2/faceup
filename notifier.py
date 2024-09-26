from slack_sdk import WebClient
from posture_detector import Postural


class SlackNotifier(Postural):
    def __init__(self, cli: WebClient, channel: str) -> None:
        super().__init__()
        self.cli = cli
        self.channel = channel

    def on_enter_face_down(self) -> None:
        # TODO attach image
        self.cli.chat_postMessage(channel=self.channel, text="face down")
