#coding=utf-8
import numpy as np
import networkx as nx
import time
import random


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, G=nx.gnm_random_graph(10000, 30000), num_runs=30):
        self.G = G
        self.p, self.q = p, q
        self.num_runs = num_runs

    def decision(self, i):  # 线性决策规则
        dose = sum([self.G.node[k]['state'] for k in self.G.node[i]['neigh']])
        prob = self.p + self.q * dose
        return True if random.random() <= prob else False

    def single_diffuse(self):  # 单次扩散
        for i in self.G.nodes_iter():
            self.G.node[i]['neigh'] = self.G.neighbors(i)
            self.G.node[i]['state'] = False

        non_adopt_set = [i for i in self.G.nodes() if not self.G.node[i]['state']]
        num_of_adopt = []
        for j in range(self.num_runs):
            x = 0
            random.shuffle(non_adopt_set)
            for i in non_adopt_set:
                if self.decision(i):
                    self.G.node[i]['state'] = True
                    non_adopt_set.remove(i)
                    x += 1
            num_of_adopt.append(x)
        return num_of_adopt

    def repete_diffuse(self, repetes=10):  # 多次扩散
        return [self.single_diffuse() for i in range(repetes)]


class Diffuse_gmm(Diffuse):
    def decision(self, i):  # gmm决策规则
        dose = sum([self.G.node[k]['state'] for k in self.G.node[i]['neigh']])
        prob = 1 - (1 - self.p) * (1 - self.q) ** dose
        return True if random.random() <= prob else False


if __name__ == '__main__':
    pq_range = [(i, j) for i in np.linspace(0.001, 0.021, num=5) for j in np.linspace(0.05, 0.145, num=20)]
    result = []
    u = 1
    for p, q in pq_range:
        t1 = time.clock()
        diff = Diffuse_gmm(p, q, num_runs=30)
        x = np.mean(diff.repete_diffuse(), axis=0)
        result.append(np.concatenate(([p, q], x)))
        print u, 'Time: %.2f s' % (time.clock() - t1)
        u += 1

    np.save('gnm_random_graph(10000, 30000)-gmm', result)