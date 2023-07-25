import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from rcl_interfaces.msg import SetParametersResult


import time
import numpy as np
import matplotlib.pyplot as plt



class RobotSimulator(Node):

    def __init__(self):
        super().__init__('MPC_Simulator')

        # Read the initial position, final setpoint Ts and prediction horizon from ROS parameters or set default values
        self.declare_parameter('initial_params', [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.1, 10.0])
        initial_params = self.get_parameter('initial_params').value
        self.initial_x = initial_params[0]
        self.initial_y = initial_params[1]
        self.initial_theta = initial_params[2]
        self.final_x = initial_params[3]
        self.final_y = initial_params[4]
        self.final_theta = initial_params[5]
        self.Ts = initial_params[6]
        self.N = initial_params[7]


        self.twist_publisher = self.create_publisher(Twist, 'robot_cmd', 10)

        # Create a subscriber for the derivative topic
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'robot_odom',
            self.odom_callback,
            10
        )



        self.x_data = []
        self.y_data = []
        self.theta_data = []



        self.x_data.append(self.initial_x)
        self.y_data.append(self.initial_y)
        self.theta_data.append(self.initial_theta)

        self.compute_and_publish_twist()




        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'initial_params':
                self.initial_params = param.value
                self.initial_x = self.initial_params[0]
                self.initial_y = self.initial_params[1]
                self.initial_theta = self.initial_params[2]
                self.final_x = self.initial_params[3]
                self.final_y = self.initial_params[4]
                self.final_theta = self.initial_params[5]
                self.Ts = self.initial_params[6]
                self.N = self.initial_params[7]
        return SetParametersResult(successful=True)


        

    def odom_callback(self, msg):
        self.state = msg.pose.pose

        self.x_data.append(self.state.position.x)
        self.y_data.append(self.state.position.y)
        self.theta_data.append(np.arcsin(self.state.orientation.z)*2)
        
        self.control = msg.twist.twist

        self.compute_and_publish_twist()

    def compute_and_publish_twist(self):
        twist = Twist()
        # call MPC function
        action = self.MPC()
        twist.linear.x = action[0] * np.cos(self.theta_data[-1])
        twist.linear.y = action[0] * np.sin(self.theta_data[-1])
        twist.angular.z = action[1]
        self.twist_publisher.publish(twist)


    def MPC(self):
        v = 0.1
        omega = 0.2
        return v, omega


def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
