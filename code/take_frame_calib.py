import cv2
import libs.lib_rs as rs

rs.window_setting("color")
d415 = rs.D415()

frames_saved = 0
path = "..\\data_2D\\circles_%d.png"
path_drawn = "..\\data_2D\\circles_drawn_%d.png"
poses_str = ""
while True:
    # key configs
    key = cv2.waitKey(50)
    if key == ord('q'):
        break
    isToSave = False
    if key == ord('s'):
        isToSave = True
    # fetch data
    is_found, color, color_drawn, pose = d415.detect_circle_board_2()
    cv2.imshow("color", color_drawn)
    # save data
    if isToSave and is_found:
        cv2.imwrite(path % frames_saved, color)
        cv2.imwrite(path_drawn % frames_saved, color_drawn)
        # infos
        print("frame%d saved" % frames_saved)
        print("pose of frame: %d" % frames_saved)
        print(pose)
        frames_saved += 1
        # add pose value into the string
        for i in range(4):
            for j in range(4):
                poses_str += ("%f " % pose[i, j])
        poses_str += '\n'

print("%d frames saved. Data:" % frames_saved)
print(poses_str)
d415.close()
