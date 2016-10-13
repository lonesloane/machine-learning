import copy
import os
import pickle
import sys
import numpy as np


def get_proximity(v1, v2):
    proximity = np.array([1 for v in v1 if v in v2], dtype=np.int8)
    return proximity.sum()


def trim_cluster(clusters, key, component):
    try:
        if key not in clusters.keys():
            return
        clusters[key].remove(component)
        if len(clusters[key]) == 0:
            del clusters[key]
    except ValueError:
        pass


class Cluster:
    def __init__(self, key):
        self.key = key
        self.length = 0
        self.mean_proximity = 0
        self.components = []
        self.features = []

    def add_component(self, key, features):
        pass


class Builder:

    def __init__(self, train=False, graph_format='graphml'):
        self.output_folder = '/home/stephane/Playground/PycharmProjects/machine-learning/Classification/' \
                             'DataPrep/output/train/'
        self.graph_format = graph_format
        self._train = train
        self.file_features = {}
        self.clusters = {}
        self.clustered = []

    def process(self):
        self.load_file_features()
        self.initial_clustering()
        self.generate_graph(graph_name='initial_clustering')
        self.trim_clusters()
        self.remove_double_edges()
        self.generate_graph(graph_name='trimmed_clusters')

    def load_file_features(self):
        with(open(os.path.join(self.output_folder, 'files_features.pkl'), 'rb')) as in_file:
            self.file_features = pickle.load(in_file)

    def initial_clustering(self):
        raise NotImplementedError("Abstract method should not be called directly.")

    def trim_clusters(self):
        raise NotImplementedError('Abstract method should not be called directly')

    def remove_double_edges(self):
        raise NotImplementedError("Abstract method should not be called directly.")

    def report(self):
        print("Report:\n")
        for file in self.clusters:
            file_cluster = np.array([v for _, v in self.clusters[file]], dtype=np.int8)
            length = file_cluster.size
            mean_proximity = file_cluster.sum() / length
            print("Cluster for %s - length: %d - mean proximity: %f" % (file, length, mean_proximity))

    def clear_cluster(self, key):
        self.clusters[key] = list()

    def add_cluster(self, key, linked_key, proximity):
        raise NotImplementedError('Abstract method should not be called directly.')

    def generate_graph(self, graph_name):
        if self.graph_format == 'graphml':
            self.generate_graphml(graph_name)
            return
        if self.graph_format == 'dot':
            self.generate_dot_graph(graph_name)
            return
        else:
            raise ValueError('Unexpected format value "%s". Supported values are "dot" and "graphml"'
                             % self.graph_format)

    def generate_graphml(self, graph_name):
        with open('output/'+graph_name+'.graphml', 'w') as graph_file:
            graph_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            graph_file.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"'
                             ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                             ' xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns"'
                             ' xmlns:y="http://www.yworks.com/xml/graphml">\n'
                             ' <key for="node" id="d0" yfiles.type="nodegraphics"/>\n'
                             ' <key for="edge" id="d1" yfiles.type="edgegraphics"/>\n'
                             '  <graph edgedefault="undirected" id="G">\n')
            for file in self.clusters:
                file_cluster = [k for k, v in self.clusters[file]]
                graph_file.write('<node id="'+file+'">\n'
                                 '<data key="d0">'
                                 '<y:ShapeNode>'
                                 '<y:Fill color="#FFCCCC" transparent="false"/>'
                                 '<y:BorderStyle color="#000000" type="line" width="1.0"/>'
                                 '<y:NodeLabel>'+file+'</y:NodeLabel>'
                                 '<y:Shape type="ellipse"/>'
                                 '</y:ShapeNode>'
                                 '</data>'
                                 '</node>\n')
                for key in file_cluster:
                    graph_file.write('<node id="'+key+'">\n'
                                     '<data key="d0">'
                                     '<y:ShapeNode>'
                                     '<y:Fill color="#FFCCCC" transparent="false"/>'
                                     '<y:BorderStyle color="#000000" type="line" width="1.0"/>'
                                     '<y:NodeLabel>' + key + '</y:NodeLabel>'
                                     '<y:Shape type="ellipse"/>'
                                     '</y:ShapeNode>'
                                     '</data>'
                                     '</node>\n')
                    graph_file.write('<edge id="'+file+'-'+key+'" directed="false" source="'+file+'" target="'+key+'">'
                                     '<data key="d1">'
                                     '<y:PolyLineEdge>'
                                     '<y:LineStyle color="#000000" type="line" width="1.0"/>'
                                     '<y:Arrows source="none" target="standard"/>'
                                     '<y:EdgeLabel></y:EdgeLabel>'
                                     '<y:BendStyle smoothed="false"/>'
                                     '</y:PolyLineEdge>'
                                     '</data>'
                                     '</edge>\n')
            graph_file.write('</graph>\n'
                             '</graphml>\n')

    def generate_dot_graph(self, graph_name):
        with open('output/'+graph_name+'.dot', 'w') as graph_file:
            graph_file.write('graph off_doc_clusters\n'
                             '{\n'
                             'graph [bgcolor=white, splines=spline, overlap=false]\n'
                             'edge [color=black]\n'
                             'node [color=rosybrown, shape=box, fontname="Verdana", fontsize=14]\n')
            for file in self.clusters:
                file_cluster = [k for k, v in self.clusters[file]]
                graph_file.write(file + ' [label="' + file + '"];\n')
                for key in file_cluster:
                    graph_file.write(key + ' [label="' + key + '", shape=ellipse, color=lightseagreen];\n')
                    graph_file.write(file + '--' + key + '\n')
            graph_file.write('}')


class BuilderClumps(Builder):
    def __init__(self, train=False, graph_format='graphml'):
        super(BuilderClumps, self).__init__(train, graph_format)

    def initial_clustering(self):
        for key, values in self.file_features.items():
            remaining_nodes = {k: v for k, v in self.file_features.items()
                               if k != key and k not in self.clustered}
            for k, v in remaining_nodes.items():
                proximity = get_proximity(values, v)
                if proximity > 0:
                    self.add_cluster(key, k, proximity)

    def add_cluster(self, key, linked_key, proximity):
        if key not in self.clusters.keys():
            self.clusters[key] = list()
        self.clusters[key].append((linked_key, proximity))

    def trim_clusters(self):
        raise NotImplementedError

    def remove_double_edges(self):
        raise NotImplementedError


class BuilderDiPoles(Builder):
    def __init__(self, train=False, graph_format='graphml'):
        super(BuilderDiPoles, self).__init__(train, graph_format)

    def initial_clustering(self):
        """
        Identify cluster roots based on maximum proximity between nodes
        :return:
        """
        for key, values in self.file_features.items():
            max_proximity = -1
            remaining_nodes = {k: v for k, v in self.file_features.items()
                               if k != key and k not in self.clustered}
            for k, v in remaining_nodes.items():
                proximity = get_proximity(values, v)
                if proximity > 0 and proximity >= max_proximity:
                    if proximity > max_proximity:
                        self.clear_cluster(key)
                    max_proximity = proximity
                    self.add_cluster(key, k, proximity)

    def add_cluster(self, key, linked_key, proximity):
        if key not in self.clusters.keys():
            self.clusters[key] = list()
        self.clusters[key].append((linked_key, proximity))

    def trim_clusters(self):
        """
        Keep only 'dipoles' by allocating nodes to their clusters of maximum proximity
        :return:
        """
        trimmed_clusters = copy.deepcopy(self.clusters)
        for key, components in self.clusters.items():
            for (linked_key, proximity) in components:
                target_proximity = get_proximity(self.file_features[key], self.file_features[linked_key])
                for other_key, other_components in self.clusters.items():
                    if other_key != linked_key:
                        continue
                    for (other_linked_key, other_proximity) in other_components:
                        if other_linked_key == key:
                            continue
                        candidate_proximity = get_proximity(self.file_features[other_key],
                                                            self.file_features[other_linked_key])
                        if candidate_proximity < target_proximity:
                            trim_cluster(trimmed_clusters, other_key, component=(other_linked_key, other_proximity))
                        elif candidate_proximity > target_proximity:
                            trim_cluster(trimmed_clusters, key, component=(linked_key, proximity))
        self.clusters = trimmed_clusters

    def remove_double_edges(self):
        """
        Initially, all nodes are clustered, which leads to duplicate edges
        :return:
        """
        trimmed_clusters = copy.deepcopy(self.clusters)
        trimmed_keys = []
        for key, components in self.clusters.items():
            for (linked_key, proximity) in components:
                for other_key, other_components in self.clusters.items():
                    if other_key != linked_key or (key, other_key) in trimmed_keys or (other_key, key) in trimmed_keys:
                        continue
                    for (other_linked_key, other_proximity) in other_components:
                        if other_linked_key != key:
                            continue
                        trim_cluster(trimmed_clusters, other_key, (other_linked_key, other_proximity))
                        trimmed_keys.append((other_key, other_linked_key))
                        trimmed_keys.append((other_linked_key, other_key))
        self.clusters = trimmed_clusters


def main():
    builder = BuilderDiPoles(train=True)
    builder.process()
    builder.report()
    print("Clusters size: %f" % len(builder.clusters))

if __name__ == '__main__':
    sys.exit(main())
