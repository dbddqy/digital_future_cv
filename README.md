# Project Setup

Environment: Win10

1. install **Anaconda**
2. create python3.7 environment in Anaconda
3. install **RealSenseSDK** and **PCL**
4. install python packages: **opencv**, **scipy**, **pyrealsense2**, **pclpy**... 
5. clone the project and configure the interpreter
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

## 5. Clone the project and configure the interpreter

### **(recommend to use PyCharm as our IDE)**
 
1. open PyCharm, choose **Get from Vision Control**  

![](/pics/02.jpg)

2. copy the URL from the github page and create the project  

![](/pics/04.jpg)
![](/pics/03.jpg)

3. go to **file->settings->Project:xxx->Project Interpreter**, click upper right sign to add new intepreter  

![](/pics/05.jpg)

4. choose Conda **Environment->Exsiting environment** and select our configured env "digital_future"  

![](/pics/06.jpg)

5. in the left project window right click **code->Mark Directory as->Sources Root**

6. later if the project is updated, just go to **VCS->update project**

## 6. Run test codes in test folder

Finally, you can run two test codes (in the testing folder).  
(for the vision one, you need to connect camera to your computer via USB3 port. And to close the program you just need to press 'q')

# Extra steps to configure the camera settings

1. open Realsense viewer and configure the Depth Unit to 0.0001 to get the highest accuracy

![](/pics/01.jpg)
