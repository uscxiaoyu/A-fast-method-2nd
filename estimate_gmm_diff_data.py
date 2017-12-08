#coding=utf-8
from estimate_bass import *

data = np.load('gnm_random_graph(10000,30000)-gmm.npy')
params_cont = []
for i, x in enumerate(data):
    t1 = time.clock()
    p, q = x[:2]
    s = x[2:]
    m_idx = np.argmax(s)
    s_in = s[: m_idx + 2]

    para_range = [[1e-6, 1e-1], [1e-4, 1], [0, 50000]]
    bass_est = Bass_estimate(x, para_range)
    bass_est.t_n = 500
    P, Q, M, R2 = bass_est.optima_search(c_n=100, threshold=1e-8)
    params_cont.append([p, q, P, Q, M, R2])
    print i + 1, 'Time elapsed: %.2f s' % (time.clock() - t1), 'R2: %.4f' % R2

np.save('estimate gnm_random_graph(10000,30000)-gmm', params_cont)