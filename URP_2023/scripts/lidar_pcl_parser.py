#!/usr/bin/env python3

import pcl
import rospy
import lidar_pcl_helper
from sensor_msgs.msg import PointCloud2

def do_voxel_grid_downssampling(pcl_data,leaf_size):
    '''
    Create a VoxelGrid filter object for a input point cloud
    :param pcl_data: point cloud data subscriber
    :param leaf_size: voxel(or leaf) size
    :return: Voxel grid downsampling on point cloud
    :https://github.com/fouliex/RoboticPerception
    '''
    vox = pcl_data.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size) # The bigger the leaf size the less information retained
    return vox.filter()

def do_passthrough(pcl_data,filter_axis,axis_min,axis_max):
    '''
    Create a PassThrough  object and assigns a filter axis and range.
    :param pcl_data: point could data subscriber
    :param filter_axis: filter axis
    :param axis_min: Minimum  axis to the passthrough filter object
    :param axis_max: Maximum axis to the passthrough filter object
    :return: passthrough on point cloud
    '''
    passthrough = pcl_data.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()


class LiDARProcessing():
    def __init__(self) -> None:
        rospy.init_node("lidar_clustering")
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback)
        self.roi_pub = rospy.Publisher("/lidar3D_ROI", PointCloud2, queue_size=1)
        self.ransac_pub = rospy.Publisher("/lidar3D_ransac", PointCloud2, queue_size=1)
        self.filt_pub = rospy.Publisher("/lidar3D_filtered", PointCloud2, queue_size=1)
        self.cluster_pub = rospy.Publisher("/lidar3D_clustered", PointCloud2, queue_size=1)

    def callback(self, data):
        cloud = lidar_pcl_helper.ros_to_pcl(data)

        # downsampling
        cloud = do_voxel_grid_downssampling(cloud,0.1)
        # x 값이 -5부터 5인 것까지 ROI
        filter_axis = 'x'
        axis_min = -5.0
        axis_max = 5.0
        cloud = do_passthrough(cloud, filter_axis, axis_min, axis_max)
        # y 값이 -5부터 5인 것까지 ROI
        filter_axis = 'y'
        axis_min = -5.0
        axis_max = 5.0
        cloud = do_passthrough(cloud, filter_axis, axis_min, axis_max)
        cloud_new = lidar_pcl_helper.pcl_to_ros(cloud)
        cloud_new.header.frame_id = "velodyne"
        self.roi_pub.publish(cloud_new)
        rospy.loginfo("Publishing!")


if __name__ == "__main__":
    try:
        LiDARProcessing()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass