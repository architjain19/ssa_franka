#!/usr/bin/env python3
import rospy

if __name__ == '__main__':
    rospy.init_node('test_python_node')
    rate = rospy.Rate(1) # 1Hz
    while not rospy.is_shutdown():
        rospy.loginfo("Test - Hello from Python!")
        rate.sleep()
