"""Python version of Negatively Correlated Search"""
import numpy as np
import problem
import ray
from ray.util.multiprocessing import Pool

class NCS_C:
    'This class contain the alogirhtm of NCS, and its API for invoking.'

    def __init__(self, Tmax, sigma, r, epoch, N, fun_num, is_distributed=False):
        self.Tmax = Tmax
        self.r = r
        self.epoch = epoch
        self.N = N
        self.D = 30
        self.fun_num = fun_num
        if is_distributed:
            self.efun = self.distributed_evalu
        else:
            self.efun = self.evalu
        self.processPool = Pool(6)
        self.bound = problem.arg(self.fun_num)
        sigma = (self.bound[1] - self.bound[0]) / 10.0
        self.sigma = np.tile(sigma, (self.N, 1))
        self.x = np.random.rand(self.N, self.D) * (self.bound[1] - self.bound[0]) + self.bound[0]
        self.fit = self.efun(self.x)
        # print(self.fit)
        pos = np.argmin(self.fit)
        self.bestfound_x = self.x[pos]
        self.bestfound_fit = self.fit[pos]

    def evalu(self, x):
        res = problem.benchmark_func3.remote(x, self.fun_num)
        return res

    def parall_evalu(self, x):
        parallPop = list(item for item in x)
        # print(parallPop)
        parallRes = self.processPool.map(self.evalu, parallPop)
        # print(parallRes)
        return np.array(parallRes)#.reshape(-1)

    def distributed_evalu(self, x):
        parall_id = [self.evalu(item) for item in x]
        parallRes = ray.get(parall_id)
        # print(parallRes)
        return np.array(parallRes)
    # def Db(self, i, j):
    #     sigma = (self.sigma[i] ** 2 + self.sigma[j] ** 2) / 2
    #     E_inv = np.identity(self.D) / sigma
    #     tmp = (1 / 2) * self.D * (np.log(sigma) - np.log(self.sigma[i] * self.sigma[j]))
    #     Db = (1 / 8) * (self.x[i] - self.x[j]) @ E_inv @ (self.x[i] - self.x[j]).T + tmp
    #     new_Db = (1 / 8) * (self.new_x[i] - self.x[j]) @ E_inv @ (self.new_x[i] - self.x[j]).T + tmp
    #     return Db, new_Db
    #
    # def Corr(self):
    #     Corrp = 1e300*np.ones(self.N)
    #     new_Corrp = 1e300 * np.ones(self.N)
    #     for i in range(self.N):
    #         arg = []
    #         for j in range(self.N):
    #             if i != j:
    #                 arg.append((i,j))
    #         res = np.array(self.processPool.starmap(self.Db, arg)).reshape(9,-1)
    #         Corrp[i] = np.min(res[:][0])
    #         new_Corrp[i] = np.min(res[:][1])
    #     return Corrp, new_Corrp
    def Corr(self, x, new_x):
        Corrp = 1e300*np.ones(self.N)
        new_Corrp = 1e300 * np.ones(self.N)
        for i in range(self.N):
            Db = 1e300*np.ones(self.N)
            new_Db = 1e300 * np.ones(self.N)
            for j in range(self.N):
                if i != j:
                    sigma = (self.sigma[i]**2 + self.sigma[j]**2) / 2
                    E_inv = np.identity(self.D) / sigma
                    tmp =  (1 / 2) * self.D * (np.log(sigma) - np.log(self.sigma[i]*self.sigma[j]))
                    Db[j] = (1 / 8) * (x[i] - x[j]) @ E_inv @ (x[i] - x[j]).T + tmp
                    new_Db[j] = (1 / 8) * (new_x[i] - x[j]) @ E_inv @ (new_x[i] - x[j]).T + tmp
            Corrp[i] = np.min(Db)
            new_Corrp[i] = np.min(new_Db)
        return Corrp, new_Corrp

    def run(self):
        t = 0
        c = np.zeros((self.N, 1))
        while t < self.Tmax:

            print(t)

            # 更新lambda t
            self.lambdat = 1.0 + np.random.randn(1) * (0.1 - 0.1 * t / self.Tmax)

            # 产生新种群x'
            new_x = self.x + self.sigma * np.random.randn(self.N, self.D)

            # 检查边界
            pos = np.where(new_x < self.bound[0])
            new_x[pos] = self.bound[0] + 0.0001
            pos = np.where(new_x > self.bound[1])
            new_x[pos] = self.bound[1] - 0.0001

            # 计算 f(x'),
            new_fit = self.efun(new_x)
            # 计算 Corr(p)和Corr(p')
            Corrp, new_Corrp = self.Corr(self.x, new_x)

            # 更新 BestFound
            pos = np.argmin(self.fit)
            if new_fit[pos] < self.bestfound_fit:
                self.bestfound_x = new_x[pos]
                self.bestfound_fit = new_fit[pos]

            # the normalization step
            norm_fit = (new_fit - self.bestfound_fit)  / (self.fit + new_fit - 2 * self.bestfound_fit)
            norm_Corrp = new_Corrp / (Corrp + new_Corrp)
            # 更新 x
            pos = np.where(norm_fit  < self.lambdat * norm_Corrp)
            self.x[pos] = new_x[pos]
            self.fit[pos] = new_fit[pos]
            c[pos] += 1
            t += 1
            #1/5 successful rule
            if t % self.epoch == 0:
                for i in range(self.N):
                    if c[i][0] > 0.2 * self.epoch:
                        self.sigma[i][0] /= self.r
                    elif c[i][0] < 0.2 * self.epoch:
                        self.sigma[i][0] *= self.r
                c = np.zeros((self.N, 1))
                # print('the {} {}'.format(t, self.bestfound_fit))
        return self.bestfound_fit

if __name__ == '__main__':
    import time
    import ray

    ray.init(num_cpus=6)
    print(ray.is_initialized())
    start_time = time.time()
    alg = NCS_C(Tmax=3000, sigma=0.0, r=0.95, epoch=10, N=100, fun_num=6, is_distributed = True)
    print(alg.run())
    print(time.time()-start_time)
    ray.shutdown()


# 试验并行版本
# import time
#
# import ray
# ray.init()
#
# RemoteNCS_C = ray.remote(NCS_C)
# Actor1 = RemoteNCS_C.remote(Tmax=30000, sigma=0.0, r=0.99, epoch=10, N=10, fun_num=13)
# Actor2 = RemoteNCS_C.remote(Tmax=30000, sigma=0.0, r=0.99, epoch=10, N=10, fun_num=13)
# Actor3 = RemoteNCS_C.remote(Tmax=30000, sigma=0.0, r=0.99, epoch=10, N=10, fun_num=13)
# Actor4 = RemoteNCS_C.remote(Tmax=30000, sigma=0.0, r=0.99, epoch=10, N=10, fun_num=13)
# start_time = time.time()
# ray.get([Actor1.run.remote(), Actor2.run.remote(), Actor3.run.remote(), Actor4.run.remote()])
# print(time.time()-start_time)