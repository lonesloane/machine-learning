import copy
import logging
import os
import pickle
import sys
import uuid

import numpy as np
from DataPrep import config, logger, logging_level

cluster_key_index = 0


def sorted_tuples(k1, k2):
    if k1 < k2:
        return (k1, k2)
    else:
        return (k2, k1)


def get_proximity(f1, f2):
    # TODO: use config file to decide which proximity...
    return get_similarity_proximity(f1, f2)
    # return euclidean_distance(f1, f2)


def euclidean_distance(f1, f2):
    # TODO: check/test this algo, doesn't do what it's supposed to
    squared = np.array([(v1-v2)**2
                        for k1, v1 in f1.items()
                        for k2, v2 in f2.items()
                        if k1 == k2],
                       dtype=np.float16)
    return squared.sum()


def get_similarity_proximity(f1, f2):
    # Todo : review definition of proximity
    proximity = 0
    for topic, weight_1 in f1.items():
        if topic in f2.keys():
            weight_2 = f2[topic]
            proximity += 1.*int(weight_1)*int(weight_2)

    unique_features = set(list(f1.keys()) + list(f2.keys()))

    if len(unique_features) > 0:
        proximity /= len(unique_features)
    else:
        proximity = 0
    return proximity


def get_clusters_proximity(cluster1, cluster2):
    total_proximity = 0
    n_bounds = 0
    # First, find keys which are common to both clusters
    # and exclude them from the calculation since their effect on the other
    # atoms has already been taken into account...
    common_keys = get_clusters_common_keys(cluster1, cluster2)
    # Next, compute proximity for each couple of atoms not in the common_keys
    for atom_key, atom_features in cluster1.atoms.items():
        if atom_key in common_keys:
            continue
        for o_atom_key, o_atom_features in cluster2.atoms.items():
            if o_atom_key in common_keys:
                continue
            total_proximity += get_proximity(atom_features, o_atom_features)
            n_bounds += 1

    cluster_proximity = total_proximity / n_bounds if n_bounds > 0 else 0
    return cluster_proximity


def get_clusters_common_keys(cluster1, cluster2):
    keys1 = frozenset(cluster1.atoms.keys())
    keys2 = frozenset(cluster2.atoms.keys())
    return keys1.intersection(keys2)


def trim_cluster(clusters, key, component, strict=True):
    try:
        if key not in clusters.keys():
            return
        clusters[key].remove(component)
        if strict and len(clusters[key]) == 0:
            del clusters[key]
    except ValueError:
        pass


class Grapher():
    def __init__(self, output_folder, current_clusters, full_clusters_list, bound_clusters=None):
        self.output_folder = output_folder
        self.current_clusters = current_clusters
        self.full_clusters_list = full_clusters_list
        self.bound_clusters = bound_clusters
        self.unique_id = 0

    def graph_components(self, graph_name):
        self.unique_id = 0
        with open(os.path.join(self.output_folder, graph_name + '.graphml'), 'w') as graph_file:
            graph_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            graph_file.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"'
                             ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                             ' xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns"'
                             ' xmlns:y="http://www.yworks.com/xml/graphml">\n'
                             ' <key for="node" id="d0" yfiles.type="nodegraphics"/>\n'
                             ' <key for="edge" id="d1" yfiles.type="edgegraphics"/>\n'
                             '  <graph edgedefault="undirected" id="G">\n')
            for component_key in self.current_clusters:
                component_key_id = self.get_id(component_key)
                # NODE
                graph_file.write(self.build_node(component_key, component_key_id))

            graph_file.write('</graph>\n'
                             '</graphml>\n')

    def build_node(self, component_key, component_key_id):
        has_inner = True if component_key in self.full_clusters_list \
                            and len(self.full_clusters_list[component_key].components.keys()) > 1 \
                            else False
        node = '<node id="' + component_key_id + '"'
        if has_inner:
            node += ' yfiles.foldertype="group"'
        node += '>\n'
        if has_inner:
            node += '<data key="d4"/>\n'
            node += '<data key="d5">\n'
            node += '<y:ProxyAutoBoundsNode>\n'
            node += '<y:Realizers active="0">\n'
            node += '<y:GroupNode>\n'
            node += '<y:Fill color="#F8ECC9" transparent="false"/>\n'
            node += '<y:Shape type="rectangle3d"/>\n'
            node += '</y:GroupNode>\n'
            node += '</y:Realizers>\n'
            node += '</y:ProxyAutoBoundsNode>\n'
            node += '</data>\n'
        node += '<data key="d0">\n'
        node += '<y:ShapeNode>\n'
        node += '<y:Fill color="#FFCCCC" transparent="false"/>\n'
        node += '<y:BorderStyle color="#000000" type="line" width="1.0"/>\n'
        node += '<y:NodeLabel>' + component_key + '</y:NodeLabel>\n'
        node += '<y:Shape type="rectangle"/>\n'
        node += '</y:ShapeNode>\n'
        node += '</data>\n'
        if has_inner:
            node += ' ' + self.inner_graph(component_key, component_key_id)
        node += '</node>\n'
        return node

    def inner_graph(self, component_key, component_key_id):
        if component_key not in self.full_clusters_list or len(self.full_clusters_list[component_key].components.keys()) <= 1:
            return ''
        inner_keys = []
        self.get_children(inner_keys, component_key)
        graph = '<graph edgedefault="undirected" id="'+component_key_id+'">\n'
        for inner_key in set(inner_keys):
            inner_key_id = self.get_id(inner_key)
            # NODE
            graph += self.build_node(inner_key, inner_key_id)
        graph += '</graph>\n'
        return graph

    def get_children(self, inner_keys, key):
        """
        Recursively get the 'atomic' components
        :param inner_keys:
        :param key:
        :return:
        """
        # if key not in self.components or len(self.components[key].components.keys()) <= 1:
        if len(self.full_clusters_list[key].components.keys()) <= 1:
            inner_keys.append(key)
            return
        keys = self.full_clusters_list[key].components.keys()
        for inner_key in keys:
            self.get_children(inner_keys, inner_key)

    def graph_clusters(self, graph_name):
        with open(os.path.join(self.output_folder, graph_name+'.graphml'), 'w') as graph_file:
            graph_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            graph_file.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"'
                             ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                             ' xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns"'
                             ' xmlns:y="http://www.yworks.com/xml/graphml">\n'
                             ' <key for="node" id="d0" yfiles.type="nodegraphics"/>\n'
                             ' <key for="edge" id="d1" yfiles.type="edgegraphics"/>\n'
                             ' <key attr.name="description" attr.type="string" for="node" id="d4"/>\n'
                             ' <key for="node" id="d5" yfiles.type="nodegraphics"/>\n'
                             ' <key attr.name="description" attr.type="string" for="edge" id="d8"/>\n'
                             ' <key for="edge" id="d9" yfiles.type="edgegraphics"/>\n'
                             '  <graph edgedefault="undirected" id="G">\n')
            for component_key in self.bound_clusters:
                component_key_id = self.get_id(component_key)
                # NODE
                graph_file.write(self.build_node(component_key, component_key_id))

                bound_keys = [k for k, _ in self.bound_clusters[component_key]]
                for bound_key in bound_keys:
                    bound_key_id = self.get_id(bound_key)

                    # NODE
                    graph_file.write(self.build_node(bound_key, bound_key_id))
                    # EDGE
                    graph_file.write(self.build_edge(component_key_id, bound_key_id))

            graph_file.write('</graph>\n'
                             '</graphml>\n')

    def build_edge(self, component_key_id, bound_key_id):
            edge = (
                '<edge id="' + component_key_id + '-' + bound_key_id + '" source="' + component_key_id
                + '" target="' + bound_key_id + '">\n'
                '</edge>\n')
            return edge

    def get_id(self, key):
        self.unique_id += 1
        return key + '::' + str(self.unique_id)


class Cluster:
    def __init__(self, components, root=False, key=None):
        self.components = components
        self.root = root
        self.key = key if key else Cluster.create_key(self.components, root)
        self.length = len(self.components.keys())
        if root:
            self.atoms = components
            self.features = components[self.key]
            self.mean_proximity = 0
        else:
            self.atoms = Cluster.process_atoms(components)
            self.features = Cluster.process_features(self.atoms)
            self.mean_proximity = Cluster.get_mean_proximity(self.atoms)

    @staticmethod
    def process_atoms(components):
        atoms = dict()
        for _, cluster in components.items():
            for key, atom in cluster.atoms.items():
                if key not in atoms:
                    atoms[key] = atom
        return atoms

    @staticmethod
    def process_features(atoms):
        features_dict = dict()
        for _, features in atoms.items():
            for topic, weight in features.items():
                if topic not in features_dict.keys():
                    features_dict[topic] = 0
                features_dict[topic] += int(weight)
        return features_dict

    @staticmethod
    def get_mean_proximity(atoms):
        upper_keys = []
        total_proximity = 0
        for key, features in atoms.items():
            proximity = 0
            for o_key, o_features in atoms.items():
                if key == o_key or o_key in upper_keys:
                    continue
                upper_keys.append(key)
                proximity += get_proximity(features, o_features)
            total_proximity += proximity
        # TODO: meaningless/useless ternary expression?
        return 0 if total_proximity == 0 else total_proximity/len(atoms)

    @staticmethod
    def create_key(components, root):
        # TODO: avoid using global variable...
        global cluster_key_index
        if root:
            for key in components.keys():
                return key.lower()
        else:
            # return str(uuid.uuid4())
            cluster_key_index += 1
            return 'cls_'+str(cluster_key_index)


class Builder:

    def __init__(self, input_data_file, min_nb_clusters=1, max_iter=100, graph_format='graphml'):
        self.output_folder = '/home/stephane/Playground/PycharmProjects/machine-learning/Classification/' \
                             'DataPrep/output/train/'
        self.input_data_file = input_data_file
        self.graph_format = graph_format
        self.min_nb_clusters = min_nb_clusters
        self.unique_id = 0
        self.max_iter = max_iter
        self.mean_clusters_proximity = 0

        self.full_clusters_list = {}
        self.current_clusters = {}
        self.previous_clusters = {}
        self.bound_clusters = {}

    def process(self):
        logger.info('=' * 20)
        logger.info('load individual components')
        self.load_individual_components()
        self.report()
        self.graph_components(graph_name='initial_components')

        iter_nb = 1
        while iter_nb <= self.max_iter and len(self.current_clusters) > self.min_nb_clusters:
            logger.info('=' * 20)
            logger.info('Iteration nb: {}'.format(iter_nb))
            logger.info('-' * 20)
            self.clusterify(iter_nb)
            logger.info('=' * 20)

            # TODO: see how best to use the total proximity to control convergence
            # As it is right now, this is too strict
            resulting_proximity = self.get_clusters_mean_proximity()
            if resulting_proximity < self.mean_clusters_proximity:
                self.current_clusters = self.previous_clusters # discard this last iteration
                self.report()
                self.graph_components(graph_name='final_clusters')
                break
            else:
                self.mean_clusters_proximity = resulting_proximity
            iter_nb += 1

        # Report final state:
        self.report_final()

    def clusterify(self, iter_nb):
        raise NotImplementedError("Abstract method should not be called directly.")

    def merge_clusters(self):
        self.previous_clusters = copy.deepcopy(self.current_clusters)
        self.current_clusters = {}
        for key, bound in self.bound_clusters.items():
            components = dict()
            components[key] = self.full_clusters_list[key]
            for bound_key, _ in bound:
                components[bound_key] = self.full_clusters_list[bound_key]
            cluster = Cluster(components, root=False)
            self.full_clusters_list[cluster.key] = cluster
            self.current_clusters[cluster.key] = cluster
        self.bound_clusters = {}

    def load_individual_components(self):
        with(open(os.path.join(self.output_folder, self.input_data_file), 'rb')) as in_file:
            individual_components = pickle.load(in_file)
            for file, features in individual_components.items():
                # TODO: extract method and add unit-test
                for key, value in features.items():
                    if value == 'N':
                        features[key] = 10
                    elif value == 'H':
                        features[key] = 100
                atom = Cluster({file.lower(): features}, root=True)
                self.full_clusters_list[atom.key] = atom
                self.current_clusters[atom.key] = atom

    def initial_clustering(self, iter_nb):
        raise NotImplementedError("Abstract method should not be called directly.")

    def get_clusters_mean_proximity(self):
        total_proximity = 0
        for key, cluster in self.current_clusters.items():
            total_proximity += cluster.mean_proximity

        return total_proximity/len(self.current_clusters)

    def remove_duplicates(self):
        inv_cluster_dict = {}
        for cluster_key, cluster in self.current_clusters.items():
            inv_key = tuple(sorted(cluster.atoms.keys()))
            if inv_key not in inv_cluster_dict:
                inv_cluster_dict[inv_key] = list()
            inv_cluster_dict[inv_key].append(cluster_key)
        for inv_key, cluster_keys in inv_cluster_dict.items():
            if len(cluster_keys) > 1:
                for cluster_key in cluster_keys[1:]:
                    self.current_clusters.pop(cluster_key)

    def trim_clusters(self):
        raise NotImplementedError('Abstract method should not be called directly')

    def remove_double_edges(self):
        raise NotImplementedError("Abstract method should not be called directly.")

    def report(self):
        logger.info('-'*20)
        logger.info("Report:")
        logger.info('-'*20)
        logger.info("Number of clusters: %d" % len(self.current_clusters))
        logger.info("Total number of components: %d" % self.nb_of_atoms())
        logger.info("Total proximity: %d" % self.get_clusters_mean_proximity())

        for cluster_key in self.current_clusters:
            logger.debug('cluster: %s - length: %s - mean proximity: %s - features: %s' %
                         (cluster_key, len(self.full_clusters_list[cluster_key].components),
                          self.full_clusters_list[cluster_key].mean_proximity, self.full_clusters_list[cluster_key].features))
            logger.debug('inner components: %s ' % self.full_clusters_list[cluster_key].components.keys())
            logger.debug('inner atoms: %s ' % self.full_clusters_list[cluster_key].atoms.keys())

        logger.info('-'*20)

    def report_final(self):
        logger.info('='*20)
        logger.info("Clusters:")
        logger.info('-'*20)
        logger.info("Number of clusters: %d" % len(self.current_clusters))
        logger.info('-'*20)

        for cluster_key in self.current_clusters:
            logger.info('cluster: %s' % cluster_key)
            logger.info('mean proximity: %s' % self.full_clusters_list[cluster_key].mean_proximity)
            logger.info('features:')
            for k,v in self.full_clusters_list[cluster_key].features.items():
                logger.info("{}: {}".format(k,v))
            logger.info('inner atoms: %s ' % self.full_clusters_list[cluster_key].atoms.keys())
            logger.info('-'*20)

        logger.info('='*20)

    def save(self):
        with open(os.path.join(self.output_folder, 'computed_clusters.pkl'), 'wb') as out:
            pickle.dump(self.current_clusters, out)

    def nb_of_atoms(self):
        atoms_list = list()
        nb_components = 0
        for cluster_key in self.current_clusters:
            atoms_list.extend(self.current_clusters[cluster_key].atoms.keys())

        return len(set(atoms_list))

    def clear_bound_clusters(self, key):
        if key in self.bound_clusters.keys():
            del self.bound_clusters[key]

    def add_bound_cluster(self, key, linked_key, proximity):
        raise NotImplementedError('Abstract method should not be called directly.')

    def graph_components(self, graph_name):
        Grapher(output_folder=self.output_folder,
                current_clusters=self.current_clusters,
                full_clusters_list=self.full_clusters_list).graph_components(graph_name=graph_name)

    def graph_clusters(self, graph_name):
        Grapher(output_folder=self.output_folder,
                current_clusters=self.current_clusters,
                full_clusters_list=self.full_clusters_list,
                bound_clusters=self.bound_clusters).graph_clusters(graph_name=graph_name)


class BuilderClumps(Builder):
    def __init__(self, input_data_file, min_nb_clusters=1, graph_format='graphml'):
        super(BuilderClumps, self).__init__(input_data_file=input_data_file,
                                            graph_format=graph_format,
                                            min_nb_clusters=min_nb_clusters)

    def initial_clustering(self):
        for key, values in self.current_clusters.items():
            remaining_nodes = {k: v for k, v in self.current_clusters.items() if k != key}
            for k, v in remaining_nodes.items():
                proximity = get_proximity(values, v)
                if proximity > 0:
                    self.add_bound_cluster(key, k, proximity)

    def add_bound_cluster(self, key, linked_key, proximity):
        if key not in self.bound_clusters.keys():
            self.bound_clusters[key] = list()
        self.bound_clusters[key].append((linked_key, proximity))

    def trim_clusters(self):
        raise NotImplementedError

    def remove_double_edges(self):
        raise NotImplementedError


class BuilderDiPoles(Builder):
    def __init__(self, input_data_file, min_nb_clusters=1, max_iter=100, graph_format='graphml'):
        super(BuilderDiPoles, self).__init__(input_data_file=input_data_file,
                                             graph_format=graph_format,
                                             min_nb_clusters=min_nb_clusters,
                                             max_iter=max_iter)
        self.orphans = None

    def clusterify(self, iter_nb):
        self.initial_clustering(iter_nb)
        if logging_level == logging.DEBUG:
            self.graph_clusters(graph_name='initial_clusters_%s' % iter_nb)
        self.remove_double_edges()
        if logging_level == logging.DEBUG:
            self.graph_clusters(graph_name='double_trimmed_clusters_%s' % iter_nb)
        self.trim_clusters()
        if logging_level == logging.DEBUG:
            self.graph_components(graph_name='trimmed_clusters_%s' % iter_nb)

        self.report()

    def initial_clustering(self, iter_nb):
        logger.info('generate bounds')
        self.generate_bounds()
        if logging_level == logging.INFO:
            self.graph_clusters(graph_name='initial_clusters_%s' % iter_nb)

        # keep only the bounds with maximum strength
        self.trim_bounds()
        logger.info("Total number of components: %d" % self.nb_of_atoms())
        logger.info('-'*30)
        if logging_level == logging.DEBUG:
            self.graph_clusters(graph_name='trimmed_bounds_clusters_%s' % iter_nb)

        logger.info('merge clusters')
        self.merge_clusters()
        logger.info("Total number of components: %d" % self.nb_of_atoms())
        logger.info('-'*30)
        if logging_level == logging.DEBUG:
            self.graph_components(graph_name='merged_clusters_%s' % iter_nb)

        logger.info('remove duplicates')
        self.remove_duplicates()
        logger.info("Total number of components: %d" % self.nb_of_atoms())
        logger.info('-'*30)
        if logging_level == logging.DEBUG:
            self.graph_components(graph_name='deduplicated_clusters_%s' % iter_nb)

        if self.orphans and len(self.orphans) > 0:
            for key in self.orphans:
                self.current_clusters[key] = self.full_clusters_list[key]

    def generate_bounds(self):
        """
        :return:
        """
        linked = list()
        for cluster_key, cluster in self.current_clusters.items():
            for linked_cluster_key, linked_cluster in self.current_clusters.items():
                if cluster_key == linked_cluster_key:
                    continue
                proximity = get_clusters_proximity(cluster, linked_cluster)
                if (len(cluster.atoms) > len(linked_cluster.atoms) and proximity > cluster.mean_proximity) \
                        or (proximity > linked_cluster.mean_proximity):
                    self.add_bound_cluster(cluster_key, linked_cluster_key, proximity)
                    linked.append(cluster_key)
                    linked.append(linked_cluster_key)

        self.orphans = frozenset(self.current_clusters.keys()).difference(frozenset(linked))

    def trim_bounds(self):
        for key, bounds in self.bound_clusters.items():
            trimmed_bounds = copy.deepcopy(bounds)
            proximities = np.array([bound[1] for bound in bounds], np.float16)
            #mean_proximity = proximities.sum() / proximities.size
            max_proximity = proximities.max()
            for bound in bounds:
                if bound[1] < 0.9 * max_proximity:
                    trimmed_bounds.remove(bound)
            self.bound_clusters[key] = trimmed_bounds

    def add_bound_cluster(self, key, linked_key, proximity):
        if key not in self.bound_clusters:
            self.bound_clusters[key] = list()
        self.bound_clusters[key].append((linked_key, proximity))

    def trim_clusters_mean_proximity(self):
        clusters_mean_proximity = self.get_clusters_mean_proximity()
        trimmed_clusters = copy.deepcopy(self.bound_clusters)
        for key, components in self.bound_clusters.items():
            for (linked_key, proximity) in components:
                for other_key, other_components in self.bound_clusters.items():
                    if other_key != linked_key:
                        continue
                    for (other_linked_key, other_proximity) in other_components:
                        if other_linked_key == key:
                            continue
                        if other_proximity < clusters_mean_proximity:
                            trim_cluster(trimmed_clusters, other_key, component=(other_linked_key, other_proximity))
                        elif other_proximity > clusters_mean_proximity:
                            trim_cluster(trimmed_clusters, key, component=(linked_key, proximity))
        self.bound_clusters = trimmed_clusters

    def trim_clusters(self):
        """
        Keep only clusters for which the proximity
        is greater than the mean proximity of the corpus
        :return:
        """
        untrimmed_keys = list()
        trimmed_keys = list()
        logger.debug('trim clusters')
        trimmed_clusters = copy.deepcopy(self.current_clusters)
        clusters_mean_proximity = self.get_clusters_mean_proximity()
        for cluster_key, cluster in self.current_clusters.items():
            if cluster.mean_proximity < clusters_mean_proximity:
                trimmed_clusters.pop(cluster_key)
                trimmed_keys.extend(cluster.components.keys())
            else:
                untrimmed_keys.extend(cluster.components.keys())

        self.current_clusters = trimmed_clusters
        orphans = set(trimmed_keys).difference(set(untrimmed_keys))
        if len(orphans)>0:
            for key in orphans:
                self.current_clusters[key] = self.full_clusters_list[key]

    def remove_double_edges(self):
        """
        Initially, all nodes are clustered, which leads to duplicate edges
        :return:
        """
        trimmed_clusters = copy.deepcopy(self.bound_clusters)
        trimmed_keys = []
        for key, components in sorted(self.bound_clusters.items()):
            for (linked_key, proximity) in components:
                for other_key, other_components in sorted(self.bound_clusters.items()):
                    if other_key != linked_key or (key, other_key) in trimmed_keys or (other_key, key) in trimmed_keys:
                        continue
                    for (other_linked_key, other_proximity) in other_components:
                        if other_linked_key != key:
                            continue
                        trim_cluster(trimmed_clusters, other_key, (other_linked_key, other_proximity), strict=False)
                        trimmed_keys.append((other_key, other_linked_key))
                        trimmed_keys.append((other_linked_key, other_key))
        self.bound_clusters = trimmed_clusters


def main():
    #input_data_file = 'files_features.pkl'
    #output_folder = '/home/stephane/Playground/PycharmProjects/machine-learning/Classification/DataPrep/output/train'

    input_data_file = 'random_files_features.pkl'
    output_folder = '/home/stephane/Playground/PycharmProjects/machine-learning/Classification/tests/data/'

    min_nb_clusters = 2
    max_iter = 10

    global cluster_key_index
    cluster_key_index = 0

    builder = BuilderDiPoles(input_data_file=input_data_file,
                             min_nb_clusters=min_nb_clusters,
                             max_iter=max_iter)
    builder.output_folder = output_folder
    builder.process()
    builder.save()

if __name__ == '__main__':
    sys.exit(main())
