import numpy as np
import ray
from ray.util.multiprocessing import Pool
import problem

def job(x ,y):
	"""
	:param x:
	:param y:
	:return:
	"""
	return x * y
if __name__ == '__main__':
    import numpy as np
    import ray
    # from ray.util.multiprocessing import Pool
    ray.init()
    print(ray.is_initialized())
    o = np.loadtxt('f6_o.txt')
    print(o)
    # processPool = Pool(4)
    # data_list = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    # res = processPool.starmap(job, data_list)
    # print(res)

    @ray.remote
    def A():
        return "A"


    @ray.remote
    def B():
        return "B"


    @ray.remote
    def C(a, b):
        return "C"


    a_id = A.remote()
    b_id = B.remote()
    c_id = C.remote(a_id, b_id)
    print(ray.get(c_id))


    ray.shutdown()
