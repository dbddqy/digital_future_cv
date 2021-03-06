import cv2
import libs.lib_rs as rs

rs.window_setting("color")
d415 = rs.D415()

frames_saved = 0
path = "..\\data_2D\\marker_%d.png"
path_drawn = "..\\data_2D\\marker_drawn_%d.png"
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
    corners, ids, color, color_drawn = d415.detect_aruco()
    cv2.imshow("color", color_drawn)
    # save data
    if isToSave:
        cv2.imwrite(path % frames_saved, color)
        cv2.imwrite(path_drawn % frames_saved, color_drawn)
        # infos
        print("frame%d saved" % frames_saved)
        frames_saved += 1

d415.close()
