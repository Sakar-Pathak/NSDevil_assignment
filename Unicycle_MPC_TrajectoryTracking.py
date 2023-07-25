import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

T = 0.1  # [s]
N = 100  # prediction horizon

v_max = 0.6
v_min = -v_max
omega_max = np.pi / 4
omega_min = -omega_max

x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.rows()

v = ca.SX.sym('v')
omega = ca.SX.sym('omega')
controls = ca.vertcat(v, omega)
n_controls = controls.rows()

rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)  # system r.h.s
f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)

U = ca.SX.sym('U', n_controls, N)  # Decision variables (controls)
P = ca.SX.sym('P', n_states + n_states)  # parameters (which include the initial state and the reference state)

X = ca.SX.sym('X', n_states, (N + 1))  # A vector that represents the states over the optimization problem.

obj = 0  # Objective function
g = []  # constraints vector

Q = np.zeros((3, 3))
Q[0, 0] = 1
Q[1, 1] = 5
Q[2, 2] = 0.1  # weighing matrices (states)

R = np.zeros((2, 2))
R[0, 0] = 0.5
R[1, 1] = 0.05  # weighing matrices (controls)

st = X[:, 0]  # initial state
g = ca.vertcat(g, st - P[:3])  # initial condition constraints

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj = obj + ca.mtimes((st - P[3:]).T, ca.mtimes(Q, (st - P[3:]))) + ca.mtimes(con.T, ca.mtimes(R, con))

    st_next = X[:, k + 1]
    f_value = f(st, con)
    st_next_euler = st + (T * f_value)
    g = ca.vertcat(g, st_next - st_next_euler)

OPT_variables = ca.vertcat(ca.reshape(X, 3 * (N + 1), 1), ca.reshape(U, 2 * N, 1))

nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

opts = {
    'ipopt.max_iter': 2000,
    'ipopt.print_level': 0,  # Set to 3 for more detailed output
    'print_time': 0,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-6
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


args = {}
# Equality constraints
args["lbg"] = np.zeros(3 * (N + 1))  # Equality constraints lower bound
args["ubg"] = np.zeros(3 * (N + 1))  # Equality constraints upper bound

# State and control constraints
args["lbx"] = np.zeros(3 * (N +1) + 2 * N)  # state x lower bound
args["ubx"] = np.zeros(3 * (N +1) + 2 * N)  # state x upper bound

# State x bounds
args["lbx"][0::3] = -2  # state y lower bound
args["ubx"][0::3] = 2  # state y upper bound

# State y bounds
args["lbx"][1::3] = -2  # state y lower bound
args["ubx"][1::3] = 2  # state y upper bound
# State theta bounds
args["lbx"][2::3] = -np.inf  # state theta lower bound
args["ubx"][2::3] = np.inf  # state theta upper bound
# Control v bounds
args["lbx"][3 * (N + 1)::2] = v_min  # v lower bound
args["ubx"][3 * (N + 1)::2] = v_max  # v upper bound
# Control omega bounds
args["lbx"][3 * (N + 1) + 1::2] = omega_min  # omega lower bound
args["ubx"][3 * (N + 1) + 1::2] = omega_max  # omega upper bound




t0 = 0
x0 = np.array([5.0, 5.0, 0.0])  # initial condition.
xs = np.array([0.0, 0.0, 0.0])  # Reference posture.

xx = []
xx.append(x0)
#xx[:, 0] = x0  # xx contains the history of states
t = [t0]   

u0 = np.zeros((N, 2))  # two control inputs for each robot
X0 = np.tile(x0, (N + 1, 1))  # initialization of the states decision variables

#print("Number of Columns:", x0.shape[1])

sim_tim = 20  # Maximum simulation time

mpciter = 0
xx1 = []
u_cl = []


ss_error = 1

# Define the shift function (Euler integration)
def shift(T, t0, x0, u, f):
    st = x0
    con = u[0, :].T
    f_value = f(st, con)
    st = st + (T * f_value)
    x0 = st.full()

    t0 = t0 + T
    u0 = np.vstack((u[1:, :], u[-1, :]))
    return t0, x0, u0

while np.linalg.norm((x0 - xs), 2) > 1e-2 and mpciter < sim_tim / T:
    args['p'] = np.concatenate((x0, xs))  # set the values of the parameters vector
    args['x0'] = np.concatenate((X0.T.reshape(3 * (N + 1), 1), u0.T.reshape(2 * N, 1)))  # initial value of the optimization variables

    sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

    u = sol['x'][3 * (N + 1):].reshape((2,N)).T  # get controls only from the solution
    #print(u[1,:])


    xx1.append(sol['x'][:3 * (N + 1)].reshape((3,N+1)).T)  # get solution TRAJECTORY
    #print(xx1)

    u_cl.append(u[:, 0])
    t0, x0, u0 = shift(T, t0, x0, u, f)

    x0 = x0.reshape(3,)

    xx.append(x0)
    X0 = sol['x'][:3 * (N + 1)].reshape((3, N + 1)).T  # get solution TRAJECTORY
    X0 = np.vstack((X0[1:, :], X0[-1, :]))


    mpciter += 1

ss_error = np.linalg.norm((x0 - xs), 2)
#average_mpc_time = main_loop_time / (mpciter + 1)

# Print the simulation results
print("Final State Error:", ss_error)


xx = np.array(xx)


# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Trajectory')

# Initialize the plot with the reference posture
ax.plot(xs[0], xs[1], 'r*', label='Reference Posture')
ax.legend()

# Initialize the arrow
arrow_length = 0.2

def update(frame):
    ax.clear()
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    #plot xx with yellow color and dotted line style
    ax.plot(xx[:, 0], xx[:, 1], 'y--', label='Robot Trajectory')
    ax.plot(xs[0], xs[1], 'r*', label='Setpoint')

    arrow = ax.arrow(xx[frame, 0], xx[frame, 1], arrow_length * np.cos(xx[frame, 2]), arrow_length * np.sin(xx[frame, 2]),
                 head_width=0.2, head_length=0.2, fc='red', ec='red')
    
    #show legend
    ax.legend()


# Create the animation
animation = FuncAnimation(fig, update, frames=len(xx[:,0]), interval=50)

plt.grid()
plt.show()
