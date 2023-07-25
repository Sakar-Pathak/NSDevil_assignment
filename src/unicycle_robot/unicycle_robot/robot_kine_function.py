import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult


class RobotKineNode(Node):

    def __init__(self):
        super().__init__('robot_kine')

        # Read the initial position and Ts from ROS parameters or set default values
        self.declare_parameter('initial_params', [0.0, 0.0, 0.0, 0.1])
        initial_params = self.get_parameter('initial_params').value
        self.initial_x = initial_params[0]
        self.initial_y = initial_params[1]
        self.initial_theta = initial_params[2]
        self.Ts = initial_params[3]

    
        self.twist_subscriber = self.create_subscription(
            Twist,
            'robot_cmd',
            self.twist_callback,
            10
        )

        # Create a publisher for the derivative vector
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/robot_odom',
            10
        )

        # Create parameter callback to update the initial values when the parameters change
    #     self.initial_params_param = self.get_parameter('initial_params')
        # Create parameter callback to update the initial values when the parameters change
        self.add_on_set_parameters_callback(self.parameter_callback)

        # publish the initial odometry
        self.twist = Twist()
        self.compute_and_publish_odometry()

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'initial_params':
                self.initial_params = param.value
                self.initial_x = self.initial_params[0]
                self.initial_y = self.initial_params[1]
                self.initial_theta = self.initial_params[2]
                self.Ts = self.initial_params[3]
        return SetParametersResult(successful=True)



    def twist_callback(self, msg):
        # Save the received control message and timestamp
        self.twist = msg
        self.compute_and_publish_odometry()


    def compute_and_publish_odometry(self):
        # Compute the derivative of the state vector for a unicycle robot using the latest messages
        x = self.initial_x + self.twist.linear.x * self.Ts
        y = self.initial_y + self.twist.linear.y * self.Ts
        theta = self.initial_theta + self.twist.angular.z * self.Ts

        x_dot = self.twist.linear.x
        y_dot = self.twist.linear.y
        theta_dot = self.twist.angular.z

        # Update the initial position
        self.initial_x = x
        self.initial_y = y
        self.initial_theta = theta


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
