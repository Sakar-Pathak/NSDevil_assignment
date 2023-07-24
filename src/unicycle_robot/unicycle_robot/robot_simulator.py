import rclpy
from rclpy.node import Node
from unicycle_robot_interfaces.msg import UnicycleState, UnicycleControl, UnicycleDerivative
import time
import numpy as np
import matplotlib.pyplot as plt

class RobotSimulator(Node):

    def __init__(self):
        super().__init__('robot_simulator')

        # Create publishers for state and control topics
        self.state_publisher = self.create_publisher(UnicycleState, 'state_vector', 10)
        self.control_publisher = self.create_publisher(UnicycleControl, 'control_vector', 10)

        # Create a subscriber for the derivative topic
        self.derivative_subscriber = self.create_subscription(
            UnicycleDerivative,
            'derivative_vector',
            self.derivative_callback,
            10
        )

        # Initialize variables for state and control
        self.state = UnicycleState()
        self.state.x = 5.0
        self.state.y = 5.0
        self.state.theta = 0.0

        self.control = UnicycleControl()
        self.control.v = 2.0
        self.control.omega = 0.2

        # Initialize variables for simulation
        self.sampling_time = 0.1  # 100 milliseconds
        self.total_time = 100.0  # 10 seconds
        self.num_steps = int(self.total_time / self.sampling_time)

        # Lists to store data for plotting
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.theta_data = []
        self.x_dot_data = []
        self.y_dot_data = []
        self.theta_dot_data = []

        self.loop_counter = 0
        # Publish the current state and control
        self.state_publisher.publish(self.state)
        self.control_publisher.publish(self.control)
        

    def derivative_callback(self, msg):
        self.derivative = UnicycleDerivative()
        # Update the control inputs based on the received derivative
        self.derivative.x_dot = msg.x_dot
        self.derivative.y_dot = msg.y_dot
        self.derivative.theta_dot = msg.theta_dot

        # Run the simulation for the specified time
        if self.loop_counter < self.num_steps:
            self.run_simulation()
            print("loop_counter: ", self.loop_counter)
            self.loop_counter += 1

        # Plot the results
        if self.loop_counter == self.num_steps:
            self.plot_results()
            plt.show()
        self.state_publisher.publish(self.state)
        self.control_publisher.publish(self.control)

    def run_simulation(self):

        # Store the data for plotting
        self.time_data.append(self.loop_counter * self.sampling_time)
        self.x_data.append(self.state.x)
        self.y_data.append(self.state.y)
        self.theta_data.append(self.state.theta)
        # Perform Euler integration to update the state
        self.state.x += self.sampling_time * self.derivative.x_dot
        self.state.y += self.sampling_time * self.derivative.y_dot
        self.state.theta += self.sampling_time * self.derivative.theta_dot
        # Store the data derivative for plotting
        self.x_dot_data.append(self.derivative.x_dot)
        self.y_dot_data.append(self.derivative.y_dot)
        self.theta_dot_data.append(self.derivative.theta_dot)


    def plot_results(self):
        # Plot the x, y and theta 
        plt.figure(1)
        plt.plot(self.time_data, self.x_data, label='x')
        plt.plot(self.time_data, self.y_data, label='y')
        plt.plot(self.time_data, self.theta_data, label='theta')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Position (meters or radians)')
        plt.title('Position vs Time')
        plt.legend()

        # Plot the x_dot, y_dot and theta_dot
        plt.figure(2)
        plt.plot(self.time_data, self.x_dot_data, label='x_dot')
        plt.plot(self.time_data, self.y_dot_data, label='y_dot')
        plt.plot(self.time_data, self.theta_dot_data, label='theta_dot')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (meters/second or radians/second)')
        plt.title('Velocity vs Time')
        plt.legend()


def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
