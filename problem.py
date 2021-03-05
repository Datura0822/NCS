import numpy as np
# import ray

def F8F2(x):
    f2 = 100 * (x[:,0]**2 - x[:,1])**2 + (1 - x[:,0])**2
    f = 1 + f2 ** 2 /4000 - np.cos(f2)
    return f

def benchmark_func1(x, fun_num):

    if fun_num == 6:
        o = np.loadtxt('f6_o.txt')
        ps, D = np.shape(x)
        if len(o) >= D:
            o = o[0:D]
        else:
            o = -90 + 180 * np.random.rand(1, D)
        x = x - np.tile(o, (ps, 1)) + 1
        f = np.sum(100*(x[:,0:D-1]**2-x[:,1:D])**2 + (x[:,0:D-1]-1)**2, axis=1)
        return f

    if fun_num == 12:
        a = np.loadtxt('f12_a.txt')
        b = np.loadtxt('f12_b.txt')
        alpha = np.loadtxt('f12_alpha.txt')
        ps, D = np.shape(x)
        if len(alpha) >= D:
            alpha = alpha[0: D]
            a = a[0: D, 0: D]
            b = b[0: D, 0: D]
        else:
            alpha = -3 + 6 * np.random.rand(1, D)
            a = np.around(-100 + 200 * np.random.rand(D, D))
            b = np.around(-100 + 200 * np.random.rand(D, D))
        alpha = np.tile(alpha, (D, 1))
        A = np.sum(a * np.sin(alpha) + b * np.cos(alpha), axis=1)
        f = np.zeros(ps)
        for i in range(ps):
            xx = np.tile(x[i,:], (D, 1))
            B = np.sum(a * np.sin(xx) + b * np.cos(xx), axis=1)
            f[i] = np.sum((A - B) ** 2, axis=0)
        return f

    if fun_num == 13:
        o = np.loadtxt('f13_o.txt')
        ps, D = np.shape(x)
        if len(o) >= D:
            o = o[0:D]
        else:
            o = -1 + 1 *np.random.rand(1, D)
        x = x - np.tile(o, (ps, 1)) + 1
        fit = 0
        for i in range(D-1):
            fit = fit + F8F2(x[:,[i, i+1]])
        fit = fit + F8F2(x[:,[D-1, 0]])
        return fit



def arg(fun_num):
    if fun_num == 6:
        bound = [-100.0, 100.0]
        return bound

    if fun_num == 12:
        bound = [-np.pi, np.pi]
        return bound

    if fun_num == 13:
        bound = [-3.0, 1.0]
        return bound


def benchmark_func2(x, fun_num):

    if fun_num == 6:
        o = np.loadtxt('f6_o.txt')
        D = len(x)
        if len(o) >= D:
            o = o[0:D]
        else:
            o = -90 + 180 * np.random.rand(D)
        x = x - o + 1
        f = np.sum(100*(x[0:D-1]**2-x[1:D])**2 + (x[0:D-1]-1)**2)
        return f

# @ray.remote
# def benchmark_func3(x, fun_num):
#
#     if fun_num == 6:
#         o = np.loadtxt('./f6_o.txt')
#         D = len(x)
#         if len(o) >= D:
#             o = o[0:D]
#         else:
#             o = -90 + 180 * np.random.rand(D)
#         x = x - o + 1
#         f = np.sum(100*(x[0:D-1]**2-x[1:D])**2 + (x[0:D-1]-1)**2)
#         return f



# x = np.random.rand(30) * 200 -100
# y = benchmark_func2(x, 6)
# print(y)
# print(y)
# o = np.loadtxt('f6_o.txt')
# x =  np.tile(o[0:30], (1,1))
# print(np.shape(x))
# ans = benchmark_func1(x, 6)
# print(ans)
# print(o)
# x1 = np.random.rand(1000,1)*5-52
# x2 = np.random.rand(1000,1)*4+78
# x = np.hstack((x1,x2))
# print(np.shape(x))
# x = np.random.rand(10000,2)*200-100
# o = np.array([81.0232000000000,-48.3950000000000,19.2316000000000])
#x = np.array([[1,2,3],[2,5,8]])
#o = np.array([1,1,1])

# f = rosenbrock_func(x,o)
# print(x,o,f)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
#
# # 数据
#
# a = x[:, 0]
# b = x[:, 1]
# c = f
#
# # 绘制散点图
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(a, b, c)
#
# # 添加坐标轴(顺序是Z, Y, X)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# plt.show()