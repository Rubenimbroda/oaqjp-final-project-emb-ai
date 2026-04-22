"""Unit tests for emotion detection output."""

import json
import unittest
from unittest.mock import patch

from EmotionDetection import emotion_detector


class DummyResponse:
    def __init__(self, text):
        self.text = text


def mocked_post(url, headers=None, json=None, timeout=None):
    text = json["raw_document"]["text"]

    emotion_map = {
        "I am glad this happened": {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.99, "sadness": 0.01},
        "I am really mad about this": {"anger": 0.99, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.01},
        "I feel disgusted just hearing about this": {"anger": 0.0, "disgust": 0.99, "fear": 0.0, "joy": 0.0, "sadness": 0.01},
        "I am so sad about this": {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.99},
        "I am really afraid that this will happen": {"anger": 0.0, "disgust": 0.0, "fear": 0.99, "joy": 0.0, "sadness": 0.01},
    }

    response_body = {"emotionPredictions": [{"emotion": emotion_map[text]}]}
    return DummyResponse(json_module.dumps(response_body))


json_module = json


class TestEmotionDetector(unittest.TestCase):
    @patch("emotion_detection.requests.post", side_effect=mocked_post)
    def test_joy(self, mocked_requests_post):
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("emotion_detection.requests.post", side_effect=mocked_post)
    def test_anger(self, mocked_requests_post):
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("emotion_detection.requests.post", side_effect=mocked_post)
    def test_disgust(self, mocked_requests_post):
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("emotion_detection.requests.post", side_effect=mocked_post)
    def test_sadness(self, mocked_requests_post):
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("emotion_detection.requests.post", side_effect=mocked_post)
    def test_fear(self, mocked_requests_post):
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()