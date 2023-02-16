# Video_data_cleaning

This Python code performs motion detection on a video stream from a camera or video file. 
It uses background subtraction and contour detection to detect motion in each frame, 
and draws a bounding box (optional and can be removed) around any detected motion. The code allows users to specify a JSON configuration file
that sets various parameters for motion detection and video recording, such as the minimum area of a detected object,
the amount of time to wait for the camera to warm up, the time to record after motion is detected, 
and the frame rate of the recorded video. The code can also save a snapshot of the first frame with motion and record 
video of the entire motion event. 
Additionally, the code provides options to display the video stream to the screen and save the recorded video to a specified directory.

It can be useful as a pre-processing stage or to record data only when there is a change in the scene.  
