#!/usr/local/bin/python3
# The motion detection algorithm in the provided code is based on comparing consecutive frames and finding the difference 
# between them. The following parameters can be used to tune the motion detection algorithm:
# camera_warmup_time: The time to wait before capturing frames after initializing the camera to allow the camera to adjust 
# to the lighting conditions.
# resizeWidth: The width of the frames to resize the input video stream to.
# delta_thresh: The threshold value used to determine whether a pixel has changed between frames.
# 'min_area: The minimum area of a contour to be considered as motion.
# 'min_motion_frames': The number of consecutive frames with motion required to trigger a recording.
#  nonMotionTimer: The number of frames to record after motion has stopped.
# create_image: Whether to save a still image of the first frame with motion.
# show_video: Whether to display the video feed on the screen.
# The motion detector output is a recording saved to a video file in the output_folder specified in the configuration file. 
# The filename of the recording includes a timestamp of when the recording started. 
# If create_image is set to True, a still image of the first frame with motion is also saved to the output_folder.

#To convert ts to mp4: ffmpeg -i Video02.ts -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k Video02.mp4

import argparse
import warnings
import datetime
import imutils
import json
import numpy as np
import os
import time
import cv2

print("[INFO] Kicking off script - " +
      datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
#ap.add_argument("-i", "--input", required=True,
#                help="path to the input video file")              
args = vars(ap.parse_args())

# filter warnings, load the configuration
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

# initialize the camera and grab a reference to the raw camera capture
# if the video argument is None, then we are reading from webcam

# Read from video file if it's specified in the configuration
if "input_video" in conf:
    camera = cv2.VideoCapture(conf["input_video"])
else:
    # Otherwise, read from the webcam
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)


#if not conf["use_ip_cam"]:
#    camera = cv2.VideoCapture(0)
#    time.sleep(0.25)

# otherwise, we are reading from a video input
#else:
#    camera = cv2.VideoCapture(conf["input_video"])

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motion_counter = 0
non_motion_timer = conf["nonMotionTimer"]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#fourcc = 0x00000020  # a little hacky, but works for now
writer = None
(h, w) = (None, None)
zeros = None
output = None
made_recording = False

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    (grabbed, frame) = camera.read()

    timestamp = datetime.datetime.now()
    motion_detected = False

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print("[INFO] Frame couldn't be grabbed. Breaking - " +
              datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=conf["resizeWidth"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        # frame.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w1, h1) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w1, y + h1), (0, 255, 0), 2)
        motion_detected = True

    fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
    record_fps = 10
    ts = timestamp.strftime("%Y-%m-%d_%H_%M_%S")
    time_and_fps = ts + " - fps: " + str(fps)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Motion Detected: {}".format(motion_detected), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, time_and_fps, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    # Check if writer is None TODO: make path configurable
    if writer is None:

        if writer is None:
          filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
          #file_path =(conf["userDir"] +"/home/tarik-devh/Projects/RecordingPEP/" "{filename}.mp4")
          file_path = os.path.join("/home/tarik-devh/Projects/RecordingPEP/Video_out", f"{filename}.mp4")
          #file_path = file_path.format(filename=filename)
        #filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        #file_path = (conf["userDir"] + "/Library/Mobile Documents/"
        #             "com~apple~CloudDocs/testMotionDetection/testing/"
        #             "{filename}.mp4")
        #file_path = file_path.format(filename=filename)

        (h2, w2) = frame.shape[:2]
        writer = cv2.VideoWriter(file_path, fourcc, record_fps, (w2, h2), True)
        zeros = np.zeros((h2, w2), dtype="uint8")

    def record_video():
        # construct the final output frame, storing the original frame
        output = np.zeros((h2, w2, 3), dtype="uint8")
        output[0:h2, 0:w2] = frame

        # write the output frame to file
        writer.write(output)
        # print("[DEBUG] Recording....")

    if motion_detected:

        # increment the motion counter
        motion_counter += 1

        # check to see if the number of frames with motion is high enough
        if motion_counter >= conf["min_motion_frames"]:
            if conf["create_image"]:
                # create image TODO: make path configurable
                #image_path = (conf["userDir"] + "/home/tarik-devh/Projects/RecordingPEP/output" "/{filename}.jpg").format(filename=filename)
                image_path = os.path.join("/home/tarik-devh/Projects/RecordingPEP/output/", f"{filename}.jpg")
                #format(filename=filename)
                cv2.imwrite(image_path, frame)

            record_video()

            made_recording = True
            non_motion_timer = conf["nonMotionTimer"]

    # If there is no motion, continue recording until timer reaches 0
    # Else clean everything up
    else:  # TODO: implement a max recording time
        # print("[DEBUG] no motion")
        if made_recording is True and non_motion_timer > 0:
            non_motion_timer -= 1
            # print("[DEBUG] first else and timer: " + str(non_motion_timer))
            record_video()
        else:
            # print("[DEBUG] hit else")
            motion_counter = 0
            if writer is not None:
                # print("[DEBUG] hit if 1")
                writer.release()
                writer = None
            if made_recording is False:
                # print("[DEBUG] hit if 2")
                os.remove(file_path)
            made_recording = False
            non_motion_timer = conf["nonMotionTimer"]

    # check to see if the frames should be displayed to screen 
    if conf["show_video"]:
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break

# cleanup the camera and close any open windows
print("[INFO] cleaning up...")
camera.release()
cv2.destroyAllWindows()