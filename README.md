# SRNN Human Motion Prediction for ROS

## Summary
This project is entirely from the paper of Ashesh Jain. In the paper, authors propose a generic and proincipled method to involved high-level spatio-temporal structures into *Recurrent Neural Networks* (*RNNs*) called *Structural-RNN* (*S-RNN*). This method has significant improvements with som spatio-temporal problems includings: human motion modeling, human-object interaction and driver maneuver anticipation. In this project, the S-RNN method is involved into ROS to make predictions of human motion.

See their project page for more infomation: [Structural-RNN: Deep Learning on Spatio-Temporal Graphs](http://asheshjain.org/srnn)

## Quickstart
### Requirements
> It is recommended to install python requirements in a virtual environment created by [conda](https://conda.io/docs/).
* ROS (Kinetic Kame on Ubuntu 16.04 or Melodic Morenia on Ubuntu 18.04)
  
  See [the official install guide](http://www.ros.org/install) to learn how to install ROS.
* Python (2.7)
* Theano (>=0.6)
* matplotlib
* Neural Models (https://github.com/asheshjain399/NeuralModels)
  
Create an virtual environment named ros_srnn in conda:
```bash
> # create env and install requirements
> conda create -n ros_srnn python=2.7 Theano matplotlib
> # activate the env
> conda activate ros_srnn
> # install rosinstall (needed if you using a conda env)
> pip install rosinstall
> # install Neural Models 
> git clone https://github.com/asheshjain399/NeuralModels.git
> cd NeuralModels
> git checkout srnn
> python setup.py develop
```

### Download dataset and pre-trained models
* [H3.6m](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)
* [pre-trained models](https://drive.google.com/drive/folders/0B7lfjqylzqmMZlI3TUNUUEFQMXc)
> You may need to create folders to to put this files.
> 
> **DATASET_PATH** will be used to refer to the path of your dataset.
> 
> **CHECKPOINTS_PATH** will be used to refer to the path of your pre-trained models.

### Build and run the demo
* Create a ROS workspace
  ```bash
  > mkdir -p ros_srnn_ws/src && cd ros_srnn_ws
  > catkin_make
  > source devel/setup.bash
  ```
* Clone the project code
  ```bash
  > git clone https://github.com/chenhaowen01/srnn_human_motion_predict_for_ros.git src/srnn_human_motion_predict_for_ros
  ```
* Build
  ```bash
  > catkin_make
  ```
* Run
  ```bash
  > # Run the predictor node, it may take very long time, wait it completely loaded. Checkpoint path could also be specified by a ros parameter called checkpoint_path.
  > roslaunch srnn_human_motion_predict_for_ros predictor.launch checkpoint_path:=CHECKPOINTS_PATH/srnn_walking/checkpoint.pik
  > # Run the publisher after the predictor node has loaded the checkpoint. You may need to run the following command with a new terminal. Dataset path could also be specified by a ros parameter called motion_dataset_path.
  > roslaunch srnn_human_motion_predict_for_ros motion_publisher.launch motion_dataset_path:=DATASET_PATH/dataset/S7/walking_1.txt
  > # Run the visualize node to see the result. You may need to run the following command with a new terminal.
  > roslaunch srnn_human_motion_predict_for_ros visulize.launch
  ```
  Then you can see rviz run, and a skeleto walking in it.
## Video
Will upload later.
## FAQ