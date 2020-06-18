# Reload Robot Base

## About data in /data_relocate_base folder  
   ![](/pics/data_relocate_base_folder.png)

1. ***b2e_ori.txt*** and ***b2e_pos.txt*** are the rotational and translational part of the robot base to end transformation, which you can read from the ABB robot.
2. ***marker_x.png*** are the photos of tags taken from different robot poses.
3. ***result.txt*** saves the result of the robot base relocation
4. ***You don't need to modify any other files.***
   
## Runing the relocation script

1. Make sure you put correct data in the ***/data_relocate_base*** folder
2. Run ***relocate_base.py***
3. The ***result.txt*** file should be updated

## Loading the result into Grasshopper

1. Use the example file ***load_robot_base.gh***
   ![](/pics/gh_path.png)
2. Modify this path to your directory of foler ***\data_locate_base***
3. You should see the result plane of markers and robot_base(marker_0 is chosen to be the world coodinate system)
   ![](/pics/relocate_base_result.png)