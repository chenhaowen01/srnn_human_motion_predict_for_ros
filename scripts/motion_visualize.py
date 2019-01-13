#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from srnn_human_motion_predict_for_ros.msg import Skeleto

import copy
import numpy as np
from srnn_human_motion_predict import data_utils
from human_motion_visualization import forward_kinematics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from human_motion_visualization import viz

g_ob = None
g_preR = np.eye(3)
g_preT = np.zeros(3)
g_parent = None
g_offset = None
g_rotInd = None
g_expmapInd = None

def motion_visualize_callback(data):
    global g_preR, g_preT, g_parent, g_offset, g_rotInd, g_expmapInd
    rospy.loginfo('%s: %s' % (data.header.seq, data.header.stamp))
    skeleto = data.skeleto
    skeleto, g_preR, g_preT = forward_kinematics.revert_coordinate_space(skeleto, g_preR, g_preT)
    xyz = forward_kinematics.fkl(skeleto, g_parent, g_offset, g_rotInd, g_expmapInd)
    rate = rospy.Rate(1000)
    while rospy.get_rostime() < data.header.stamp:
        rate.sleep()
    g_ob.update(xyz)
    plt.draw()

def visualize():
    global g_ob, g_parent, g_offset, g_rotInd, g_expmapInd
    g_parent, g_offset, g_rotInd, g_expmapInd = forward_kinematics._some_variables()

    rospy.init_node('motion_visualize', anonymous=True)
    rospy.Subscriber('motion', Skeleto, motion_visualize_callback, queue_size=100)

    plt.show(block=True)

if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig)
    g_ob = viz.Ax3DPose(ax)
    try:
        visualize()
    except rospy.ROSInterruptException:
        pass