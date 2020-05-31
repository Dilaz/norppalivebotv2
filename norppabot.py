import argparse
import datetime
import imutils
import time
from datetime import datetime
import cv2
import numpy as np
import os
from videoreader import VideoReader
import tensorflow as tf
import json
import tweepy
import random

THRESHOLD = 0.55
MODEL_FILE = "models/norppabot.tflite"
INPUT_MEAN = 127.5
INPUT_STD = 127.5
WIDTH = 1280
HEIGHT = 720


MESSAGES_FILE = 'messages.json'
TWITTER_CONFIG = 'twitter_config.json'
TWEET_TIME_MINUTES = 30

class Detector:
    def __init__(self, camera):
        self.initModel()
        self.initTwitter()
        self.frameTime = 0
        self.frames = 0
        self.lastImageTime = 0
        self.fps = 30
        self.lastTweetTime = 0
        self.vr = VideoReader(camera, self.fps)

    def initModel(self):
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.model_height = self.input_details[0]['shape'][1]
        self.model_width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

    def initTwitter(self):
        with open(TWITTER_CONFIG) as f:
            data = json.load(f)
        auth = tweepy.OAuthHandler(data['apiKey'], data['apiSecret'])
        auth.set_access_token(data['token'], data['tokenSecret'])
        self.twitter = tweepy.API(auth)

    def detect(self, image):
        image = image.copy()

        # Test model on random input data.
        input_shape = self.input_details[0]['shape']
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.model_width, self.model_height))
        input_data = np.expand_dims(image_resized, axis=0)
        if self.floating_model:
            input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run the network
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        return boxes, scores, classes

    def triggerDetection(self, frame, orig_frame):
        self.log('Norppa detected!')

        self.lastDetectTime = time.time()

        # Only react every 30 seconds
        if (time.time() - self.lastImageTime) < 30:
            return

        # Save the original image and frame with detection boxes
        filename = "images/frame%s.jpg" % datetime.now().strftime("%Y%m%d-%H%M%S")
        o_filename = "images/orig_frame%s.jpg" % datetime.now().strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(filename, frame)
        cv2.imwrite(o_filename, orig_frame)
        self.log("Wrote %s" % filename)

        # Tweet
        self.tweetNorppaIsLive(frame)

        self.lastImageTime = time.time()

    def tweetNorppaIsLive(self, frame):
        # only tweet every TWEET_TIME_MINUTES minutes
        if time.time() - self.lastTweetTime < TWEET_TIME_MINUTES * 60:
            return

        # Get random message from messages.json
        with open(MESSAGES_FILE, 'r') as f:
            messages = json.load(f)

        randomMessage = random.choices(messages)[0].replace('#', 'hashtag-')

        filename = 'uploaded_image.jpg'
        cv2.imwrite(filename, frame)
        self.twitter.update_with_media(filename, status=randomMessage)
        self.log('Tweeted: %s' % randomMessage)
        os.remove(filename)
        self.lastTweetTime = time.time()

    def checkCoordinates(self, x, y, x2, y2):
        # You can add coordinate & width/height checks here
        width = x2 - x
        height = y2 - y
        area = width * height
        return area < 350000 and area > 10000

    def log(self, msg):
        date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(date, msg)

    def run(self):
        print("Ready")
        self.vr.start()
        self.startTime = time.time()
        frameCounter = 0
        lastFrameTime = 0
        detected = False
        detections = 0

        # Wait for the stream to start
        while not self.vr.more() or time.time() - self.startTime > 10:
            time.sleep(0.5)

        frame = None
        while True:
            if not self.vr.more():
                print('asd')
                break

            # Read the latest frame
            frame = np.copy(self.vr.read())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameCounter += 1

            lastFrameTime = time.time()


            orig_frame = frame.copy()
            detected = False

            # Get detection boxes and scores from tensorflow
            boxes, scores, classes = self.detect(frame)

            # Debug logging
            if scores[0] > 0.4:
                self.log(scores[0])

            # Go through the boxes
            for i in range(len(scores)):
                score = scores[i]
                box = boxes[i]

                y = int(box[0] * HEIGHT)
                x = int(box[1] * WIDTH)
                y2 = int(box[2] * HEIGHT)
                x2 = int(box[3] * WIDTH)

                if score < THRESHOLD:
                    continue

                # Some sanity checks
                if not self.checkCoordinates(x,y,x2,y2):
                    continue

                # Norppa detected here!
                detected = True
                self.log(score)
                print((x,y), (x2,y2))

                # Draw the detection box to the frame
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                label = '%s: %d%%' % ('norppa', int(score * 100))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(y, labelSize[1] + 10)
                # Draw white box to put label text in
                cv2.rectangle(frame, (x, label_ymin-labelSize[1]-10), (
                    x+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if detected:
                detections += 1
            else:
                detections = 0

            # Only detect after the second frame in a row that breaks the threshold
            if detections >= 2:
                self.triggerDetection(frame, orig_frame)

            if frameCounter % 1000 == 0:
                self.log('1000 frames')
                frameCounter = 0
            frameTime = time.time() - lastFrameTime
            if frameTime < 1 / self.fps:
                time.sleep(1 / self.fps - frameTime)
        self.vr.stop()
        time.sleep(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    args = vars(ap.parse_args())

    video = args["video"]

    detector = Detector(video)
    detector.run()


if __name__ == "__main__":
    main()

