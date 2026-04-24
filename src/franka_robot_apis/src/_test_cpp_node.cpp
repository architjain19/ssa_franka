#include <ros/ros.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_cpp_node");
    ros::NodeHandle nh;
    ros::Rate rate(1);
    while (ros::ok()) {
        ROS_INFO("Test - Hello from C++!");
        rate.sleep();
    }
    return 0;
}
