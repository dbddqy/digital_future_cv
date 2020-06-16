import cv2

path_color = "..\\data_2D\\wood_color_%d.png"
path_depth = "..\\data_2D\\wood_depth_%d.png"

depth = cv2.imread(path_depth % 0, cv2.IMREAD_UNCHANGED)
print(depth.shape)
