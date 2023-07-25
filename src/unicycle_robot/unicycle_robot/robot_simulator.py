import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from rcl_interfaces.msg import SetParametersResult


import time
import numpy as np
import matplotlib.pyplot as plt

import casadi as ca
from matplotlib.animation import FuncAnimation



class RobotSimulator(Node):

    def __init__(self):
        super().__init__('MPC_Simulator')

        # Read the initial position, final setpoint Ts and prediction horizon from ROS parameters or set default values
        self.declare_parameter('initial_params', [0.0, 0.0, 0.0, 0.1, 10.0])
        initial_params = self.get_parameter('initial_params').value
        self.final_x = initial_params[0]
        self.final_y = initial_params[1]
        self.final_theta = initial_params[2]
        self.Ts = initial_params[3]
        self.N = initial_params[4]

        #convert self.N to int
        self.N = int(self.N)

        # robot parameters
        self.v_max = 0.6
        self.v_min = -self.v_max
        self.omega_max = np.pi / 4
        self.omega_min = -self.omega_max

        # MPC initialization
        self.MPC_init()

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


        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'initial_params':
                self.final_x = self.initial_params[0]
                self.final_y = self.initial_params[1]
                self.final_theta = self.initial_params[2]
                self.Ts = self.initial_params[3]
                self.N = self.initial_params[4]
        #convert self.N to int
        self.N = int(self.N)
        return SetParametersResult(successful=True)


        

    def odom_callback(self, msg):
        self.state = msg.pose.pose

        self.x_data.append(self.state.position.x)
        self.y_data.append(self.state.position.y)
        self.theta_data.append(np.arcsin(self.state.orientation.z)*2)
        
        self.control = msg.twist.twist

        self.initial_x = self.x_data[-1]
        self.initial_y = self.y_data[-1]
        self.initial_theta = self.theta_data[-1]

        #print latest data
        #self.get_logger().info('x: %f, y: %f, theta: %f' % (self.initial_x, self.initial_y, self.initial_theta))
        
        self.compute_and_publish_twist()

        if np.linalg.norm(np.array([self.final_x, self.final_y, self.final_theta]) - np.array([self.initial_x, self.initial_y, self.initial_theta])) < 0.1:
            self.get_logger().info('Goal Reached!')
            self.visualize()

    def compute_and_publish_twist(self):
        twist = Twist()

        # call MPC function
        action = self.MPC()

        #print action
        print(action[0], action[1])

        twist.linear.x = action[0] * np.cos(self.theta_data[-1])
        twist.linear.y = action[0] * np.sin(self.theta_data[-1])
        twist.angular.z = action[1]
        self.twist_publisher.publish(twist)

    def MPC_init(self):
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.rows()

        self.v = ca.SX.sym('v')
        self.omega = ca.SX.sym('omega')
        self.controls = ca.vertcat(self.v, self.omega)
        self.n_controls = self.controls.rows()

        self.rhs = ca.vertcat(self.v * ca.cos(self.theta), self.v * ca.sin(self.theta), self.omega)  # system r.h.s
        self.f = ca.Function('f', [self.states, self.controls], [self.rhs])  # nonlinear mapping function f(x,u)

        self.U = ca.SX.sym('U', self.n_controls, self.N)  # Decision variables (controls)
        self.P = ca.SX.sym('P', self.n_states + self.n_states)  # parameters (which include the initial state and the reference state)

        self.X = ca.SX.sym('X', self.n_states, (self.N + 1))  # A vector that represents the states over the optimization problem.

        self.obj = 0  # Objective function
        self.g = []  # constraints vector

        self.Q = np.zeros((3, 3))
        self.Q[0, 0] = 1
        self.Q[1, 1] = 5
        self.Q[2, 2] = 0.1  # weighing matrices (states)

        self.R = np.zeros((2, 2))
        self.R[0, 0] = 0.5
        self.R[1, 1] = 0.05  # weighing matrices (controls)

        self.st = self.X[:, 0]  # initial state
        self.g = ca.vertcat(self.g, self.st - self.P[:3])  # initial condition constraints

        for k in range(self.N):
            self.st = self.X[:, k]
            con = self.U[:, k]
            self.obj = self.obj + ca.mtimes((self.st - self.P[3:]).T, ca.mtimes(self.Q, (self.st - self.P[3:]))) + ca.mtimes(con.T, ca.mtimes(self.R, con))

            st_next = self.X[:, k + 1]
            f_value = self.f(self.st, con)
            st_next_euler = self.st + (self.Ts * f_value)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)

        self.OPT_variables = ca.vertcat(ca.reshape(self.X, 3 * (self.N + 1), 1), ca.reshape(self.U, 2 * self.N, 1))

        self.nlp_prob = {'f': self.obj, 'x': self.OPT_variables, 'g': self.g, 'p': self.P}

        self.opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,  # Set to 3 for more detailed output
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)
        
        
        self.args = {}
        # Equality constraints
        self.args["lbg"] = np.zeros(3 * (self.N + 1))  # Equality constraints lower bound
        self.args["ubg"] = np.zeros(3 * (self.N + 1))  # Equality constraints upper bound
        
        # State and control constraints
        self.args["lbx"] = np.zeros(3 * (self.N +1) + 2 * self.N)  # state x lower bound
        self.args["ubx"] = np.zeros(3 * (self.N +1) + 2 * self.N)  # state x upper bound
        
        # State x bounds
        self.args["lbx"][0::3] = -10  # state y lower bound
        self.args["ubx"][0::3] = 10  # state y upper bound
        
        # State y bounds
        self.args["lbx"][1::3] = -10  # state y lower bound
        self.args["ubx"][1::3] = 10  # state y upper bound
        # State theta bounds
        self.args["lbx"][2::3] = -np.inf  # state theta lower bound
        self.args["ubx"][2::3] = np.inf  # state theta upper bound
        # Control v bounds
        self.args["lbx"][3 * (self.N + 1)::2] = self.v_min  # v lower bound
        self.args["ubx"][3 * (self.N + 1)::2] = self.v_max  # v upper bound
        # Control omega bounds
        self.args["lbx"][3 * (self.N + 1) + 1::2] = self.omega_min  # omega lower bound
        self.args["ubx"][3 * (self.N + 1) + 1::2] = self.omega_max  # omega upper bound

        
        self.u0 = np.zeros((self.N, 2))  # two control inputs for each robot

        #print("Number of Columns:", x0.shape[1])

        self.sim_tim = 20  # Maximum simulation time
        
        self.MPC_initialized = False
                    
        
    def MPC(self):

        self.x0 = np.array([self.initial_x, self.initial_y, self.initial_theta])  # initial condition.
        self.xs = np.array([self.final_x, self.final_y, self.final_theta])  # Reference posture.

        if not self.MPC_initialized:
            self.X0 = np.tile(self.x0, (self.N + 1, 1))  # initialization of the states decision variables
            MPC_initialized = True

        self.args['p'] = np.concatenate((self.x0, self.xs))  # set the values of the parameters vector
        self.args['x0'] = np.concatenate((self.X0.T.reshape(3 * (self.N + 1), 1), self.u0.T.reshape(2 * self.N, 1)))  # initial value of the optimization variables

        sol = self.solver(x0=self.args['x0'], lbx=self.args['lbx'], ubx=self.args['ubx'], lbg=self.args['lbg'], ubg=self.args['ubg'], p=self.args['p'])

        u = sol['x'][3 * (self.N + 1):].reshape((2,self.N)).T  # get controls only from the solution

        self.X0 = sol['x'][:3 * (self.N + 1)].reshape((3, self.N + 1)).T  # get solution TRAJECTORY
        self.X0 = np.vstack((self.X0[1:, :], self.X0[-1, :]))

        # initialize the control solution for the next optimization step
        self.u0 = np.vstack((u[1:, :], u[-1, :]))


        v = float(u[0, 0])
        omega = float(u[0, 1])
        return v, omega
    
    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-1, 6)
        self.ax.set_ylim(-1, 6)
        #plot xx with yellow color and dotted line style
        self.ax.plot(self.x_data[:], self.y_data[:], 'y--', label='Robot Trajectory')
        self.ax.plot(self.xs[0], self.xs[1], 'r*', label='Setpoint')

        arrow = self.ax.arrow(self.x_data[frame], self.y_data[frame], self.arrow_length * np.cos(self.theta_data[frame]), self.arrow_length * np.sin(self.theta_data[frame]),
                     head_width=0.2, head_length=0.2, fc='red', ec='red')

        #show legend
        self.ax.legend()


    def visualize(self):

        # Create a figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Robot Trajectory')

        # Initialize the plot with the reference posture
        self.ax.plot(self.xs[0], self.xs[1], 'r*', label='Reference Posture')
        self.ax.legend()

        # Initialize the arrow
        self.arrow_length = 0.2

        # Create the animation
        animation = FuncAnimation(self.fig, self.update, frames=len(self.x_data), interval=50)

        plt.grid()
        plt.show()



def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
