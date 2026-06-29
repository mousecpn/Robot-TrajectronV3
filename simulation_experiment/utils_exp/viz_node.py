#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import torch
from spatialmath import SE3
from utils_exp.transform import Rotation, Transform, SO3_R3
import struct
import time
import geometry_msgs.msg
import std_msgs.msg
import matplotlib.colors
from tf2_ros import TransformBroadcaster
DELETE_MARKER_MSG = Marker(action=Marker.DELETEALL)
DELETE_MARKER_ARRAY_MSG = MarkerArray(markers=[DELETE_MARKER_MSG])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("RedGreen", ["r", "g"])

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width

class TrajectoryVisualizationNode(Node):
    def __init__(self):
        super().__init__('trajectory_visualization_node')
        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        
        # QoS profile for visualization
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # Publishers
        self.predicted_path_pub = self.create_publisher(
            Path, '/predicted_trajectory', qos_profile, callback_group=self.callback_group)
        self.current_path_pub = self.create_publisher(
            Path, '/current_trajectory', qos_profile, callback_group=self.callback_group)
        self.grasp_markers_pub = self.create_publisher(
            MarkerArray, '/grasp_poses', qos_profile, callback_group=self.callback_group)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/scene_pointcloud', qos_profile, callback_group=self.callback_group)
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/end_effector_pose', qos_profile, callback_group=self.callback_group)
        self.target_grasp_pub = self.create_publisher(
            PoseStamped, '/target_grasp', qos_profile, callback_group=self.callback_group)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer for periodic publishing
        self.timer = self.create_timer(0.05, self.publish_visualizations, callback_group=self.callback_group)
        
        # Data storage
        self.predicted_trajectories = []
        self.current_trajectory = []
        self.grasp_poses = []
        self.pointcloud_data = None
        self.current_ee_pose = None
        self.target_grasp_pose = None
        
        self.get_logger().info('Trajectory Visualization Node initialized')

    def update_predicted_trajectory(self, predictions, current_pose, relative=False):
        """
        Update predicted trajectory from Trajectron output
        Args:
            predictions: torch.Tensor of shape [num_samples, timesteps, 4, 4]
            current_pose: 4x4 transformation matrix of current end-effector pose
        """
        self.predicted_trajectories = []
        if predictions is not None:
            current_transform = torch.tensor(current_pose, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
            predictions = torch.tensor(predictions, dtype=torch.float32)
            if relative:
                predictions = current_transform @ predictions
            predictions = torch.cat((current_transform.repeat(predictions.shape[0], 1, 1, 1), predictions), dim=1)
            self.predicted_trajectories = list(predictions.numpy())

            # Convert predictions to poses
            # for sample_idx in range(min(3, predictions.shape[0])):  # Visualize top 3 samples
            #     trajectory = []
                
            #     trajectory = torch.tensor(predictions[sample_idx], dtype=torch.float32)
                
            #     for t in range(predictions.shape[1]):
            #         # Get relative transformation from log-map
            #         relative_pose = SO3_R3().exp_map(predictions[sample_idx, t]).to_matrix()
            #         # Apply to current pose
            #         next_pose = current_transform @ relative_pose
            #         trajectory.append(next_pose.numpy())
            #         current_transform = next_pose
                
            #     self.predicted_trajectories.append(trajectory)

    def update_current_trajectory(self, ee_trajectory):
        """
        Update current trajectory history
        Args:
            ee_trajectory: list or array of 4x4 transformation matrices
        """
        self.current_trajectory = ee_trajectory
    
    def draw_grasps(self, grasps, scores, finger_depth):
        markers = []
        for i, (grasp, score) in enumerate(zip(grasps, scores)):
            msg = self._create_grasp_marker_msg(grasp, score, finger_depth)
            msg.id = i
            markers.append(msg)
        msg = MarkerArray(markers=markers)
        self.grasp_markers_pub.publish(msg)
    
    def _create_grasp_marker_msg(self, grasp, score, finger_depth):
        radius = 0.1 * finger_depth
        w, d = grasp.width, finger_depth
        scale = [radius, 0.0, 0.0]
        color = cmap(float(score))
        msg = self._create_marker_msg(Marker.LINE_LIST, "fr3_link0", grasp.pose, scale, color)
        msg.points = [to_point_msg(point) for point in _gripper_lines(w, d)]
        return msg
    
    def clear(self,):
        self.update_pointcloud(np.zeros((0,3)))
        self.grasp_markers_pub.publish(DELETE_MARKER_ARRAY_MSG)


    

    def _create_marker_msg(self, marker_type, frame, pose, scale, color):
        msg = Marker()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type = marker_type
        msg.action = Marker.ADD
        msg.pose = to_pose_msg(pose)
        msg.scale = to_vector3_msg(scale)
        msg.color = to_color_msg(color)
        return msg

    def update_grasp_poses(self, grasp_poses, grasp_scores=None):
        """
        Update grasp pose candidates
        Args:
            grasp_poses: torch.Tensor of shape [N, 4, 4] - grasp transformation matrices
            grasp_scores: torch.Tensor of shape [N] - grasp quality scores (optional)
        """
        self.grasp_poses = []
        for i, pose in enumerate(grasp_poses):
            # score = grasp_scores[i] if grasp_scores is not None else 1.0
            self.grasp_poses.append(
                Grasp(Transform.from_matrix(pose), width=0.08)
            )
        if len(self.grasp_poses) > 0:
            self.draw_grasps(self.grasp_poses, grasp_scores, finger_depth=0.05)

    def update_pointcloud(self, pointcloud):
        """
        Update scene pointcloud
        Args:
            pointcloud: numpy array of shape [N, 3] - 3D points
        """
        if pointcloud is not None:
            self.pointcloud_data = pointcloud
            pc_msg = self.create_pointcloud_msg(self.pointcloud_data)
            self.pointcloud_pub.publish(pc_msg)

    def update_current_ee_pose(self, pose):
        """
        Update current end-effector pose
        Args:
            pose: 4x4 transformation matrix
        """
        self.current_ee_pose = pose

    def update_target_grasp(self, grasp_pose):
        """
        Update target grasp pose
        Args:
            grasp_pose: 4x4 transformation matrix
        """
        self.target_grasp_pose = grasp_pose


    def create_grasp_markers(self):
        """Create marker array for grasp poses"""
        marker_array = MarkerArray()
        
        for i, grasp_info in enumerate(self.grasp_poses):
            pose_matrix = grasp_info['pose']
            score = grasp_info['score']
            
            # Create gripper visualization marker
            marker = Marker()
            marker.header.frame_id = "fr3_link0"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "grasps"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(pose_matrix[0, 3])
            marker.pose.position.y = float(pose_matrix[1, 3])
            marker.pose.position.z = float(pose_matrix[2, 3])
            
            # Orientation
            quat = self.rotation_matrix_to_quaternion(pose_matrix[:3, :3])
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])
            
            # Scale based on grasp score
            scale = 0.02 + 0.08 * score  # Scale between 0.02 and 0.1
            marker.scale.x = scale
            marker.scale.y = scale * 0.3
            marker.scale.z = scale * 0.3
            
            # Color based on score (green = good, red = bad)
            marker.color.r = 1.0 - score
            marker.color.g = score
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker.lifetime.sec = 0  # Persistent
            marker_array.markers.append(marker)
            
        return marker_array

    def create_pointcloud_msg(self, points):
        """Convert numpy pointcloud to PointCloud2 message"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "fr3_link0"
        
        # Define point cloud fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Convert points to bytes
        cloud_data = []
        for point in points:
            cloud_data.append(struct.pack('fff', point[0], point[1], point[2]))
        
        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # 3 * 4 bytes
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.data = b''.join(cloud_data)
        cloud_msg.is_dense = True
        
        return cloud_msg

    def create_pose_stamped_msg(self, pose_matrix, frame_id="fr3_link0"):
        """Convert 4x4 matrix to PoseStamped message"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        
        # Position
        pose_msg.pose.position.x = float(pose_matrix[0, 3])
        pose_msg.pose.position.y = float(pose_matrix[1, 3])
        pose_msg.pose.position.z = float(pose_matrix[2, 3])
        
        # Orientation
        quat = self.rotation_matrix_to_quaternion(pose_matrix[:3, :3])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
        return pose_msg

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            
        return np.array([x, y, z, w])

    def publish_visualizations(self):
        """Publish all visualization messages"""
        # time.sleep(0.1)  # Small delay to avoid overloading
        try:
            # Publish predicted trajectories
            for i, trajectory in enumerate(self.predicted_trajectories):
                if len(trajectory) > 0:
                    path_msg = self.create_path_msg(trajectory, ee_frame_id="future_link{}".format(i))
                    self.predicted_path_pub.publish(path_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing future visualizations: {str(e)}')
        try:
            # Publish current trajectory
            if len(self.current_trajectory) > 0:
                path_msg = self.create_path_msg(self.current_trajectory, ee_frame_id="cur_link")
                self.current_path_pub.publish(path_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing future visualizations: {str(e)}')
            
            # Publish grasp markers
            # if len(self.grasp_poses) > 0:
            #     marker_array = self.create_grasp_markers()
            #     self.grasp_markers_pub.publish(marker_array)
            
            # Publish pointcloud
            # if self.pointcloud_data is not None:
            #     pc_msg = self.create_pointcloud_msg(self.pointcloud_data)
            #     self.pointcloud_pub.publish(pc_msg)
            
            # Publish current end-effector pose
            if self.current_ee_pose is not None:
                pose_msg = self.create_pose_stamped_msg(self.current_ee_pose)
                self.ee_pose_pub.publish(pose_msg)
            
            # Publish target grasp pose
            # if self.target_grasp_pose is not None:
            #     pose_msg = self.create_pose_stamped_msg(self.target_grasp_pose)
            #     self.target_grasp_pub.publish(pose_msg)
                

        
    
    def create_path_msg(self, trajectory, frame_id="fr3_link0", ee_frame_id="cur_link", color_alpha=1.0):
        """Convert trajectory to ROS Path message"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        
        for pose_matrix in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            
            # Extract position
            pose_stamped.pose.position.x = float(pose_matrix[0, 3])
            pose_stamped.pose.position.y = float(pose_matrix[1, 3])
            pose_stamped.pose.position.z = float(pose_matrix[2, 3])
            
            # Extract orientation (convert rotation matrix to quaternion)
            rot_matrix = pose_matrix[:3, :3]
            quat = self.rotation_matrix_to_quaternion(rot_matrix)
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])
            
            path_msg.poses.append(pose_stamped)

        if trajectory is not None and len(trajectory) > 0:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "fr3_link0"        # 父坐标系
            t.child_frame_id = ee_frame_id       # 你想要的末端执行器frame名字

            # 填充位置
            t.transform.translation.x = float(pose_matrix[0, 3])
            t.transform.translation.y = float(pose_matrix[1, 3])
            t.transform.translation.z = float(pose_matrix[2, 3])

            # 填充姿态 (旋转矩阵转四元数)
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])

            self.tf_broadcaster.sendTransform(t)
                    
        return path_msg



def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def to_point_msg(position):
    """Convert numpy array to a Point message."""
    msg = geometry_msgs.msg.Point()
    msg.x = position[0]
    msg.y = position[1]
    msg.z = position[2]
    return msg


def from_point_msg(msg):
    """Convert a Point message to a numpy array."""
    return np.r_[msg.x, msg.y, msg.z]


def to_vector3_msg(vector3):
    """Convert numpy array to a Vector3 message."""
    msg = geometry_msgs.msg.Vector3()
    msg.x = vector3[0]
    msg.y = vector3[1]
    msg.z = vector3[2]
    return msg


def from_vector3_msg(msg):
    """Convert a Vector3 message to a numpy array."""
    return np.r_[msg.x, msg.y, msg.z]


def to_quat_msg(orientation):
    """Convert a `Rotation` object to a Quaternion message."""
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def from_quat_msg(msg):
    """Convert a Quaternion message to a Rotation object."""
    return Rotation.from_quat([msg.x, msg.y, msg.z, msg.w])


def to_pose_msg(transform):
    """Convert a `Transform` object to a Pose message."""
    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(transform.translation)
    msg.orientation = to_quat_msg(transform.rotation)
    return msg


def to_transform_msg(transform):
    """Convert a `Transform` object to a Transform message."""
    msg = geometry_msgs.msg.Transform()
    msg.translation = to_vector3_msg(transform.translation)
    msg.rotation = to_quat_msg(transform.rotation)
    return msg


def from_transform_msg(msg):
    """Convert a Transform message to a Transform object."""
    translation = from_vector3_msg(msg.translation)
    rotation = from_quat_msg(msg.rotation)
    return Transform(rotation, translation)


def to_color_msg(color):
    """Convert a numpy array to a ColorRGBA message."""
    msg = std_msgs.msg.ColorRGBA()
    msg.r = color[0]
    msg.g = color[1]
    msg.b = color[2]
    msg.a = color[3] if len(color) == 4 else 1.0
    return msg


def to_cloud_msg(points, intensities=None, frame=None, stamp=None):
    """Convert list of unstructured points to a PointCloud2 message.

    Args:
        points: Point coordinates as array of shape (N,3).
        colors: Colors as array of shape (N,3).
        frame
        stamp
    """
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = stamp

    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = False

    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    msg.point_step = 12
    data = points

    if intensities is not None:
        msg.fields.append(PointField("intensity", 12, PointField.FLOAT32, 1))
        msg.point_step += 4
        data = np.hstack([points, intensities])

    msg.row_step = msg.point_step * points.shape[0]
    msg.data = data.astype(np.float32).tostring()

    return msg



def _gripper_lines(width, depth):
    return [
        [0.0, 0.0, -depth / 2.0],
        [0.0, 0.0, 0.0],
        [0.0, -width / 2.0, 0.0],
        [0.0, -width / 2.0, depth],
        [0.0, width / 2.0, 0.0],
        [0.0, width / 2.0, depth],
        [0.0, -width / 2.0, 0.0],
        [0.0, width / 2.0, 0.0],
    ]
if __name__ == '__main__':
    main()