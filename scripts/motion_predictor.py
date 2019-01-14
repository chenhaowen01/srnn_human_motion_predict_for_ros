#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import Header
from srnn_human_motion_predict_for_ros.msg import Skeleto

import copy
import numpy as np
from srnn_human_motion_predict import data_utils
from srnn_human_motion_predict.motion_predict import Predictor

g_prefix_sequence_length = rospy.get_param('prefix_sequence_length', 50)
g_predicted_sequence_length = rospy.get_param('predicted_sequence_length', 50)
g_checkpoint_path = ''
g_skeleto_buffer = np.zeros((g_predicted_sequence_length, 99))
g_skeleto_index = 0
g_predictor = None
g_pub = None
g_preSeq = 0

def predict(sequence, start_time):
    # data_utils.writeFloatAsCVS('gt%s.txt' % rospy.get_rostime(), sequence)
    predicted_motion_skeleto = g_predictor.predict(sequence, g_predicted_sequence_length)
    # data_utils.writeFloatAsCVS('pd%s.txt' % rospy.get_rostime(), predicted_motion_skeleto)
    
    header = Header()
    header.frame_id = 'main'
    skeleto = Skeleto()
    skeleto.header = header
    seq = 0
    for motion_skeleto in predicted_motion_skeleto:
        skeleto.header.seq = seq
        skeleto.header.stamp = start_time + rospy.Duration.from_sec(rospy.get_param('frames_interval', 0.02)) * seq
        skeleto.skeleto = motion_skeleto
        g_pub.publish(skeleto)
        seq += 1

def motion_skeleto_subscriber_callback(data):
    global g_skeleto_buffer, g_skeleto_index, g_preSeq
    rospy.loginfo('%s: %s' % (data.header.seq, data.header.stamp))
    if data.header.seq < g_preSeq:
        rospy.loginfo('invalid seq!')
        g_preSeq = 0
        g_skeleto_index = 0

    g_skeleto_buffer[:-1, :] = g_skeleto_buffer[1:, :]
    g_skeleto_buffer[-1, :] = data.skeleto
    g_preSeq = data.header.seq
    stamp = data.header.stamp

    g_skeleto_index += 1

    if g_skeleto_index >= g_prefix_sequence_length and g_skeleto_index % g_predicted_sequence_length == 0:
        rospy.loginfo('make a prediction...')
        predict(copy.deepcopy(g_skeleto_buffer), stamp + rospy.Duration.from_sec(rospy.get_param('frames_interval', 0.02)))

def motion_predictor():
    global g_checkpoint_path, g_predictor, g_pub

    print('creating predictor...')
    g_checkpoint_path = rospy.get_param('checkpoint_path', g_checkpoint_path)
    g_predictor = Predictor(g_checkpoint_path)

    rospy.init_node('motion_predictor')
    rospy.Subscriber('motion_skeleto', Skeleto, motion_skeleto_subscriber_callback)
    g_pub = rospy.Publisher('predicted_motion_skeleto', Skeleto, queue_size=g_predicted_sequence_length)
    rospy.spin()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        g_checkpoint_path = sys.argv[1]
    try:
        motion_predictor()
    except rospy.ROSInterruptException:
        pass