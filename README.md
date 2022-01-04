# Vision
 

# Usage:
1. git the packge into your working space

    ```
    cd catkin_ws/src
    git clone https://github.com/shaluraju/Vision.git
    cd ..
    catkin_make
    ```
2. roslaunch the detect_orientation launch file:
    ```
    source devel/setup.bash
    roslaunch detect_orientation detect_orientation.launch

    ```

3. You need to have a camera that is publishing the camerage image to the topic "/camera/left/image_raw" or else change the topic name accordingly
