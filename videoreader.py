from threading import Thread
import sys
import time
import numpy as np
import ffmpeg

width = 1280
height = 720


class VideoReader:
    def __init__(self, path, fps):
        self.stopped = False
        self.frame = None
        self.path = path
        self.fps = fps
        self.process = None
        self.frames = 0
        self.firstFrame = False

    def start(self):
        # Start ffmpeg output and read it from stdout
        self.process = (
            ffmpeg
            .input(self.path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # Get the latest frame from ffmpeg and save it
        currentTime = 0
        lastFrameTime = 0
        while True:
            in_bytes = self.process.stdout.read(width * height * 3)
            if not in_bytes:
                self.stop()
                return
            self.firstFrame = True
            frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
            )

            self.frame = frame
            self.frames += 1
            frameTime = time.time() - lastFrameTime
            if lastFrameTime != 0 and frameTime < 1/self.fps:
                time.sleep(max(1/self.fps - frameTime, 0))
            lastFrameTime = time.time()


    def read(self):
        return self.frame

    def more(self):
        return self.stopped == False and self.firstFrame


    def stop(self):
        print('STOP'*10)
        self.stopped = True
        self.process.stdout.close()
        self.process.wait()


