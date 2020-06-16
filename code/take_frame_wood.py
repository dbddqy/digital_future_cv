import libs.lib_rs as rs
import cv2

rs.window_setting("color+depth")
d415 = rs.D415()

path_color = "..\\data_2D\\wood_color_%d.png"
path_depth = "..\\data_2D\\wood_depth_%d.png"

frames_saved = 0
while True:
    # key configs
    key = cv2.waitKey(50)
    if key == ord('q'):
        break
    isToSave = False
    if key == ord('s'):
        isToSave = True
    # fetch data
    color, depth = d415.get_frames()
    cv2.imshow("color", color)
    cv2.imshow("depth", depth)
    if isToSave:
        cv2.imwrite(path_color % frames_saved, color)
        cv2.imwrite(path_depth % frames_saved, depth)
        # infos
        print("frame%d saved" % frames_saved)
        frames_saved += 1
