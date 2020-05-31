# Norppalivebot v2

This is a Twitter bot that tries to detect Saimaa ringed seals from the finnish WWF livestream: https://luontolive.wwf.fi/norppa

## Setup
Download the tflite model from https://drive.google.com/open?id=1oTW4eGU6HfHfm8sEZ1pR-S3gj1iJNtYa and place it in `models/` -directory.

```pip install -r requirements.txt```

To use GPU accelerated detection, you need to install `tensorflow-gpu` instead and follow the instructions in https://www.tensorflow.org/install/gpu

To use Coral Edge TPU devices, see https://coral.ai/docs/edgetpu/tflite-python/

### Twitter config
Copy `twitter_config.json.example` to `twitter_config.json` and add your own Twitter API configs there

### Messages
You can add your own random messages to `messages.json`

## Usage
I recommend using [youtube-dl](https://ytdl-org.github.io/youtube-dl/index.html) to get the YouTube stream url:

```
python3 norppabot.py -v `youtube-dl -g -f 95 https://youtu.be/HV6JtqBmVfU`
```

To run detection for a video, you can use. Please note the video is not visible by default and you need to use OpenCV2 to show it.
```
python3 norppabot.py -v test_video.mp4
```

## Training your own mode
I used [Google Cloud AutoML Vision](https://cloud.google.com/automl/) to train the network, but you can also use [YOLO](https://github.com/AlexeyAB/darknet) and turn it into tflite model using https://github.com/hunglc007/tensorflow-yolov4-tflite
Using YOLO requires some changes to the code since it returns the detection boxes in a different format.