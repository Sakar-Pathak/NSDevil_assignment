import rclpy
from rclpy.node import Node
import numpy as np

from unicycle_robot_interfaces.msg import UnicycleState, UnicycleControl
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

class RobotKineNode(Node):

    def __init__(self):
        super().__init__('robot_kine')

        # create subscriptions to the topics
        self.state_subscriber = self.create_subscription(
            UnicycleState,
            'state_vector',
            self.state_callback,
            10
        )

        self.control_subscriber = self.create_subscription(
            UnicycleControl,
            'control_vector',
            self.control_callback,
            10
        )

        # Create a publisher for the derivative vector
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/robot_odom',
            10
        )

        # Initialize variables to store the latest state and control messages and their timestamps
        self.latest_state_msg = None
        self.latest_control_msg = None
        self.latest_state_timestamp = None
        self.latest_control_timestamp = None


        # Sampling time of the robot (adjust as needed)
        self.sampling_time = 1  # 1seconds

        # Time threshold for synchronization (adjust as needed)
        self.time_threshold = self.sampling_time * 0.5  # 50% of sampling time

    def state_callback(self, msg):
        # Save the received state message and timestamp
        self.latest_state_msg = msg
        self.latest_state_timestamp = self.get_clock().now()

        # If both state and control messages are available and synchronized, compute the derivative and publish
        if self.are_messages_synchronized():
            self.compute_and_publish_odometry()

    def control_callback(self, msg):
        # Save the received control message and timestamp
        self.latest_control_msg = msg
        self.latest_control_timestamp = self.get_clock().now()

        # If both state and control messages are available and synchronized, compute the derivative and publish
        if self.are_messages_synchronized():
            self.compute_and_publish_odometry()

    def are_messages_synchronized(self):
        # Check if both state and control messages are available and have valid timestamps
        if (
            self.latest_state_msg is not None and
            self.latest_control_msg is not None and
            self.latest_state_timestamp is not None and
            self.latest_control_timestamp is not None
        ):
            # Get the timestamps and compare using the desired time threshold (e.g., 100 milliseconds)
            state_timestamp = self.latest_state_timestamp
            control_timestamp = self.latest_control_timestamp

            time_difference = abs((state_timestamp - control_timestamp).nanoseconds * 1e-9)
            return time_difference < self.time_threshold

        return False

    def compute_and_publish_odometry(self):
        # Compute the derivative of the state vector for a unicycle robot using the latest messages
        state_vector = [self.latest_state_msg.x, self.latest_state_msg.y, self.latest_state_msg.theta]
        control_vector = [self.latest_control_msg.v, self.latest_control_msg.omega]

        x, y, theta = state_vector
        v, omega = control_vector

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        # Pack the derivative values into a custom message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'


        
            # Set the pose information (position and orientation)
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0  # Assuming a 2D environment, so z is set to 0
    
        # Set the orientation (quaternion representation)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = np.sin(theta / 2)
        odom_msg.pose.pose.orientation.w = np.cos(theta / 2)

        # set the covariance for pose to zero
        # cov_x = 0.0
        # cov_y = 0.0
        # cov_z = 0.0
        # cov_roll = 0.0
        # cov_pitch = 0.0
        # cov_yaw = 0.0
        # odom_msg.pose.covariance = [cov_x, 0, 0, 0, 0, 0,
        #                     0, cov_y, 0, 0, 0, 0,
        #                     0, 0, cov_z, 0, 0, 0,
        #                     0, 0, 0, cov_roll, 0, 0,
        #                     0, 0, 0, 0, cov_pitch, 0,
        #                     0, 0, 0, 0, 0, cov_yaw]
        odom_msg.pose.covariance = [0.0]*36
        
        odom_msg.twist.twist.linear.x = x_dot
        odom_msg.twist.twist.linear.y = y_dot
        odom_msg.twist.twist.linear.z = 0.0

        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = theta_dot
        # set the covariance for twist to zero
        odom_msg.twist.covariance = [0.0]*36

        # Publish the derivative message
        self.odom_publisher.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotKineNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
