import pclpy
# import numpy as np

cloud = pclpy.pcl.PointCloud.PointXYZ()
point = pclpy.pcl.point_types.PointXYZ()

for _ in range(10):
    point.x += 1.
    point.y += 2.
    point.z += 3.
    cloud.points.append(point)

print(cloud.xyz)
