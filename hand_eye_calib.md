# Hand-Eye Calibration Instructions

1. Prepare the calibration board circles pattern  
[link](https://github.com/opencv/opencv/blob/master/doc/acircles_pattern.png)  
**(make sure the distance between two centroids is precisely 4cm)**

2. Mount the camera  
**(make sure it does't move at all)**

3. Generate some (ideally more than 50) poses for the robot to reach. **In those poses the camera should be able to capture the entire calibration board.**  
Save all the poses into a .txt file in 4*4 matrix representation, every pose takes one line (see examples in folder ***data_calib***)

4. Run take_frame_calib.py  
When it detects, it should show something like this:  
![](/pics/hand_eye_01.png)  
Press 's' to save frames. Finally, press 'q' to quit. It should print out all poses (in matrix representation in one line) of saved frame.  
![](/pics/hand_eye_02.png)  
Save it into a .txt file

5. You can find saved frames in foler ***data_2D***. (Theoretically we don't need them, just for backup)
