import numpy as np
import math
import matplotlib.pyplot as plt
import imageio

# 车辆模型
class Vehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

    def update(self, a, omega, dt):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.v += a * dt

class Trajectory:
    def __init__(self):
        self.cx = np.linspace(0, 50, 500)
        self.cy = [np.sin(ix / 3.0) * ix / 2.0 for ix in self.cx]
        self.theta = [np.arctan2(self.cy[i+1] - self.cy[i], self.cx[i+1] - self.cx[i]) if i < len(self.cx)-1 else 0.0 for i in range(len(self.cx))]
        self.v = np.full_like(self.cx, 3.0)

    def get_reference(self, index):
        return np.array([self.cx[index], self.cy[index], self.theta[index], self.v[index]])
    
class iLQRTController:
    def __init__(self, N=50, max_iter=10, dt=0.1):
        self.N = N
        self.max_iter = max_iter
        self.dt = dt
        self.Q = np.diag([0.1, 1.0, 0.5, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.Qf = self.Q * 10

    def ilqr(self, vehicle, trajectory, index):
        x_dim = 4
        u_dim = 2
        xs = np.zeros((self.N + 1, x_dim))
        us = np.zeros((self.N, u_dim))
        xs[0] = np.array([vehicle.x, vehicle.y, vehicle.theta, vehicle.v])
        us = np.zeros((self.N, u_dim))

        for iteration in range(self.max_iter):
            for k in range(self.N):
                xs[k+1]=self.dynamics(xs[k], us[k], self.dt)

            Vx = self.Qf @ (xs[-1] - trajectory.get_reference(index + self.N))
            Vxx = self.Qf
            d_list = []
            K_list = []
            for k in reversed(range(self.N)):
                xk = xs[k]
                uk = us[k]
                x_ref = trajectory.get_reference(index + k)
                fx, fu = self.linearize_dynamics(xk, uk, self.dt)
                lx = self.Q @ (xk - x_ref)
                lu = self.R @ uk
                lxx = self.Q
                luu = self.R
                lux = np.zeros((u_dim, x_dim))
                Qx = lx + fx.T @ Vx
                Qu = lu + fu.T @ Vx
                Qxx = lxx + fx.T @ Vxx @ fx
                Quu = luu + fu.T @ Vxx @ fu
                Qux = lux + fu.T @ Vxx @ fx
                ss_test = np.eye(u_dim).tolist()
                ss_Quu = Quu.tolist()
                Quu_inv = np.linalg.inv(Quu + np.eye(u_dim) * 1e10)
                d = -Quu_inv @ Qu
                K = -Quu_inv @ Qux
                Vx = Qx + k.T @ Quu @ k + k.T @ Qu + Qux.T @ k
                Vxx = Qxx + k.T @ Quu @ k + k.T @ Qux + Qux.T @ k
                d_list.insert(0, d)
                K_list.insert(0, K)
            x_new = np.copy(xs[0])
            xs_new = [x_new]
            us_new = []
            for k in range(self.N):
                du = k_list[k] + K_list[k] @ (x_new - xs[k])
                us_new.append(us[k] + du)
                x_new = self.dynamics(x_new, us_new[-1], self.dt)
                xs_new.append(x_new)
            xs = np.array(xs_new)
            us = np.array(us_new)
            cost = self.compute_total_cost(xs, us, trajectory, index)
            print(f"Iteration: {iteration}, Cost: {cost}")
            if cost < 1e-6:
                break
            return us[0]
        
    def dynamics(self, x, u, dt):
        """
        动力学模型
        """
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + x[3] * np.cos(x[2]) * dt  # x
        x_next[1] = x[1] + x[3] * np.sin(x[2]) * dt  # y
        x_next[2] = x[2] + u[1] * dt  # theta
        x_next[3] = x[3] + u[0] * dt  # v
        return x_next

    def linearize_dynamics(self, x, u, dt):
        fx = np.eye(4)
        fx[0, 2] = -x[3] * np.sin(x[2]) * dt
        fx[0, 3] = np.cos(x[2]) * dt
        fx[1, 2] = x[3] * np.cos(x[2]) * dt
        fx[1, 3] = np.sin(x[2]) * dt
        fx[2, 2] = 1.0
        fu = np.zeros((4, 2))
        fu[2, 1] = dt
        fu[3, 0] = dt
        return fx, fu

    def compute_total_cost(self, xs, us, trajectory, index):
        cost = 0.0
        for k in range(self.N):
            xk = xs[k]
            uk = us[k]
            x_ref = trajectory.get_reference(index + k)
            dx = xk - x_ref
            cost += dx.T @ self.Q @ dx + uk.T @ self.R @ uk
        dx = xs[-1] - trajectory.get_reference(index + self.N)
        cost += dx.T @ self.Qf @ dx
        return cost
    
def main():
    vehicle = Vehicle()
    trajectory = Trajectory()
    controller = iLQRTController(N=50, max_iter=10, dt=0.1)
    dt = 0.1
    x_history = []
    y_history = []
    total_time = len(trajectory.cx) - controller.N - 1
    fig, ax = plt.subplots()
    frames = []

    for t in range(total_time):
        u_opt = controller.ilqr(vehicle, trajectory, t)
        vehicle.update(u_opt[0], u_opt[1], dt)
        x_history.append(vehicle.x)
        y_history.append(vehicle.y)

        ax.cla()
        ax.plot(trajectory.cx, trajectory.cy, "-r", label="Reference Trajectory")
        ax.plot(x_history, y_history, "-b", label="Vehicle Trajectory")
        ax.set_xlim(0, 50)
        ax.set_ylim(-20, 25)
        ax.set_title(f"iLQR Trajectory Tracking - Step {t}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.pause(0.001)

    imageio.mimsave('/home/sanmu/CILQR_Motion_Planning/test/ilqr_trajectory_tracking.gif', frames, fps=10)
    plt.show()

if __name__ == "__main__":
    main()