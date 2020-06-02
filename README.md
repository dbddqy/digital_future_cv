# Project Setup

Environment: Win10

1. install **Anaconda**
2. create python3.7 environment in Anaconda
3. install **RealSenseSDK** and **PCL**
4. install python packages: **opencv**, **scipy**, **pyrealsense2**, **pclpy**... 
5. download the python codes, create the project and configure the interpreter
6. run test codes in test folder

## 1. Install Anaconda2 (2 or 3 doesn't matter)

[Anaconda](https://www.anaconda.com/products/individual)

## 2. Create python3.7 environment (see cheatsheet)

Open Anaconda prompt, use following commands,

to list out the all the environments:
```
conda env list
```

to create new environment (e.g., named digital_future):
```
conda create --name digital_future python=3.7
```

to activate our environment:
```
conda activate digital_future
```

to check installed packages of the current activated environment:
```
conda list
```

to install a package (e.g., scipy):
```
conda install scipy
```


## 3. Install RealSenseSDK and PCL

[RealSenseSDK](https://github.com/IntelRealSense/librealsense/releases/download/v2.35.0/Intel.RealSense.SDK-WIN10-2.35.0.1758.exe)  
[PCL](https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.9.1/PCL-1.9.1-AllInOne-msvc2017-win64.exe)

## 4. Install python packages: **opencv**, **scipy**, **pyrealsense2**, **pclpy**...

opencv:
```
conda install -c menpo opencv
```

scipy:
```
conda install scipy
```

pyrealsense2:
```
pip install pyrealsense2
```
 
pclpy:
```
conda install -c conda-forge -c davidcaron pclpy
```

## 5. Download the python codes, create the project and configure the interpreter

# Extra steps configuring the camera settings

[]("\pics\01.jpg")