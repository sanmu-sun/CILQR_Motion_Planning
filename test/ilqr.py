import numpy as np

# 定义系统动态方程
def f(x, u):
    # 这里是一个简单的示例，实际应用中需要根据具体系统定义
    return x + u

# 定义运行代价函数
def l(x, u):
    return 0.5 * (x**2 + u**2)

# 定义终端代价函数
def l_f(x):
    return 0.5 * x**2

# iLQR算法实现
def ilqr(N, x0, max_iter=100, epsilon=1e-6):
    # 初始化控制序列
    u = np.zeros(N)
    # 初始化状态序列
    x = np.zeros(N + 1)
    x[0] = x0
    for k in range(N):
        x[k + 1] = f(x[k], u[k])

    for iter in range(max_iter):
        # 保存当前代价函数值
        J = l_f(x[-1]) + np.sum([l(x[k], u[k]) for k in range(N)])

        # 线性化和二次化
        A = np.ones(N)
        B = np.ones(N)
        Q = np.ones(N)
        R = np.ones(N)
        H = np.zeros(N)
        q = x[:-1]
        r = u
        q_N = x[-1]
        Q_N = 1

        # 反向传播
        V = np.zeros(N + 1)
        v = np.zeros(N + 1)
        K = np.zeros(N)
        k = np.zeros(N)
        V[-1] = Q_N
        v[-1] = q_N
        for k_idx in range(N - 1, -1, -1):
            Q_xx = Q[k_idx] + A[k_idx]**2 * V[k_idx + 1]
            Q_xu = H[k_idx] + A[k_idx] * B[k_idx] * V[k_idx + 1]
            Q_uu = R[k_idx] + B[k_idx]**2 * V[k_idx + 1]
            Q_x = q[k_idx] + A[k_idx] * v[k_idx + 1]
            Q_u = r[k_idx] + B[k_idx] * v[k_idx + 1]
            K[k_idx] = -Q_xu / Q_uu
            k[k_idx] = -Q_u / Q_uu
            V[k_idx] = Q_xx + K[k_idx]**2 * Q_uu
            v[k_idx] = Q_x + K[k_idx] * Q_uu * k[k_idx]

        # 正向传播
        alpha = 1.0
        x_new = np.zeros(N + 1)
        u_new = np.zeros(N)
        x_new[0] = x0
        for k_idx in range(N):
            delta_x = x_new[k_idx] - x[k_idx]
            delta_u = k[k_idx] + K[k_idx] * delta_x
            u_new[k_idx] = u[k_idx] + alpha * delta_u
            x_new[k_idx + 1] = f(x_new[k_idx], u_new[k_idx])

        # 计算新的代价函数值
        J_new = l_f(x_new[-1]) + np.sum([l(x_new[k], u_new[k]) for k in range(N)])

        # 判断收敛
        if np.abs(J_new - J) < epsilon:
            break

        x = x_new
        u = u_new

    return x, u

# 示例参数
N = 10
x0 = 1.0
x_opt, u_opt = ilqr(N, x0)
print("Optimal state sequence:", x_opt)
print("Optimal control sequence:", u_opt)