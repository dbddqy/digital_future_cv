import numpy as np
import pclpy
import libs.lib_rs_old as rs
import cv2
import libs.lib_frame as opt

path_color = "..\\..\\data_3D\\wood_color_%d.png"
path_depth = "..\\..\\data_3D\\wood_depth_%d.png"

rs.window_setting("color")
d415 = rs.D415()
color = cv2.imread(path_color % 0)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
depth = cv2.imread(path_depth % 0, cv2.IMREAD_UNCHANGED)

# =====================================
# image based method
# =====================================
corners = []
count = 0


def select_point(event, x, y, flags, param):
    global count, corners
    if count < 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            corner = np.array([[[x, y]]], dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
            cv2.cornerSubPix(gray, corner, (3, 3), (-1, -1), criteria)
            u = corner.flatten()[0]
            v = corner.flatten()[1]
            cv2.circle(color, (u, v), 10, [0, 0, 255], thickness=2)
            corners.append([u, v])
            if count >= 1:
                cv2.line(color, (corners[count-1][0], corners[count-1][1]), (corners[count][0], corners[count][1]), [255, 0, 0], thickness=2)
            if count == 3:
                cv2.line(color, (corners[count][0], corners[count][1]), (corners[0][0], corners[0][1]), [255, 0, 0], thickness=2)
            print("x: %3.3f, y: %3.3f" % (u, v))
            count += 1
            print(corners)
            print(count)


cv2.setMouseCallback("color", select_point)

while True:
    # key configs
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("color", color)

# w2b [4, 4]
w2b = np.loadtxt("..\\..\\data_locate_base\\result.txt").reshape([4, 4])

# b2e [4, 4]
b2e_pos = np.loadtxt("..\\..\\data_locate_base\\b2e_pos.txt") / 1000.
b2e_ori = np.loadtxt("..\\..\\data_locate_base\\b2e_ori.txt")
b2e = opt.t_4_4s__quats(opt.swap_quats(b2e_ori), b2e_pos)[0]

# e2c [4, 4]
c2e = np.load("..\\..\\data_locate_base\\c2e.npy")
e2c = opt.inv(c2e)


points_3d = np.zeros([4, 3])
for i in range(4):
    u = corners[i][0]
    v = corners[i][1]
    point_c = np.ones([4, 1])
    point_c[0:3, :] = d415.construct_point(u, v, 0.1*depth[int(round(v)), int(round(u))]).reshape([3, 1])
    # point_w = w2b.dot(b2e).dot(e2c).dot(point_c)
    point_w = point_c
    points_3d[i] = point_w[0:3].reshape([3, ])
np.savetxt("..\\..\\data_locate_base\\wood.txt", points_3d)
# =====================================
# point cloud based method
# =====================================

# cloud = []
#
# for v in range(depth.shape[0]):
#     if v % 10 != 0:
#         continue
#     for u in range(depth.shape[1]):
#         if u % 10 != 0:
#             continue
#         d = depth[v, u] * 0.1
#         if d == 0.:
#             continue
#         cloud.append(d415.construct_point(u, v, d))
#
# cloud_np = np.array(cloud)
# np.savetxt("cloud.txt", cloud_np, fmt="%8.3f")
