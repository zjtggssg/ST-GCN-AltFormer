import numpy as np
from graph import tools

num_node = 46
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1),(1, 2),(1, 3),(1, 19),(2, 3),(2, 19),(3, 4),(4, 5),(5, 6),(3, 7),(7, 8),(8, 9),(9, 10),(7, 11),(11, 12),
 (12, 13),(13, 14),(11, 15),(15, 16),
 (16, 17),
 (17, 18),
 (15, 19),
 (19, 20),
 (20, 21),
 (21, 22),
 (23, 24),
 (24, 25),
 (24, 26),
 (24, 42),
 (25, 26),
 (25, 42),
 (26, 27),
 (27, 28),
 (28, 29),
 (26, 30),
 (30, 31),
 (31, 32),
 (32, 33),
 (30, 34),
 (34, 35),
 (35, 36),
 (36, 37),
 (34, 38),
 (38, 39),
 (39, 40),
 (40, 41),
 (38, 42),
 (42, 43),
 (43, 44),
 (44, 45)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A


def main():
    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']
    np.set_printoptions(threshold=np.nan)
    for m in mode:
        print('=' * 10 + m + '=' * 10)
        print(Graph(m).get_adjacency_matrix())


if __name__ == '__main__':
    main()