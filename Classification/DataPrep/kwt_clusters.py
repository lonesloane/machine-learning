import os
import pickle
import pprint
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from DataPrep import logger
from DataPrep.clusterbuilder import Cluster

matplotlib.rcParams['font.size'] = 8.0
output_folder = 'output/train'
fig = plt.figure()


def report_final(clusters):
    logger.info('=' * 20)
    logger.info("Clusters:")
    logger.info('-' * 20)
    logger.info("Number of clusters: %d" % len(clusters))
    logger.info('-' * 20)

    for cluster_key in clusters:
        logger.info('cluster: %s' % cluster_key)
        logger.info('nb of file: %s' % len(clusters[cluster_key].atoms.keys()))
        logger.info('mean proximity: %s' % clusters[cluster_key].mean_proximity)
        logger.debug('inner atoms: %s ' % clusters[cluster_key].atoms.keys())
        logger.debug('features:')
        for k, v in clusters[cluster_key].features.items():
            logger.debug("{}: {}".format(k, v))
        logger.debug('-' * 20)

        logger.info('=' * 20)


def load_mappings():
    with open(os.path.join(output_folder, 'clusters_lvl_I.pkl'), 'rb') as input_file:
        mappings = pickle.load(input_file)
    logger.info('{} mappings loaded.'.format(len(mappings)))
    logger.debug('mappings: \n{}'.format(pprint.pformat(mappings)))
    return mappings


def load_file_features():
    file_features = {}
    with open(os.path.join(output_folder, 'files_features.pkl'), 'rb') as input_file:
        f_features = pickle.load(input_file)
    for jt_file,details in f_features.items():
        file_features[jt_file] = {}
        for topic, weight in details.items():
            file_features[jt_file][topic] = 100 if weight == 'H' else 10
    logger.info('{} file features loaded.'.format(len(file_features)))
    logger.debug('file features: \n{}'.format(pprint.pformat(file_features)))
    return file_features


def generate_clusters(mappings, file_features):
    missing_files = []
    full_clusters = {}
    for kwt, jt_files in mappings.items():
        # if kwt != 'KWT-1': continue
        logger.info('cluster {}: {} files from mappings: \n{}'.format(kwt, len(jt_files), jt_files))
        clusters = {}
        components = {}
        for f in jt_files:
            if f not in file_features.keys():
                missing_files.append(f)
                continue
            components[f.lower()] = file_features[f]
            logger.debug('adding {}'.format(f.lower()))
            logger.debug('components {}'.format(components))
            cluster = Cluster(components=components, root=True)
            components = {}
            logger.debug('cluster: {} - {}'.format(cluster.key, cluster.components))
            clusters[cluster.key] = cluster
            logger.debug('clusters: \n{}'.format(clusters))
        kwt_cluster = Cluster(components=clusters, key=kwt, root=False)

        logger.debug('cluster {}'.format(kwt_cluster.key))
        logger.debug('components: \n{}'.format(pprint.pformat(kwt_cluster.components)))
        logger.debug('length: {}'.format(kwt_cluster.length))
        logger.debug('mean proximity: {}'.format(kwt_cluster.mean_proximity))
        logger.debug('features: \n{}'.format(pprint.pformat(kwt_cluster.features)))

        # logger.info(pprint.pformat(components))
        logger.info('missing files: \n{}'.format(pprint.pformat(missing_files)))
        full_clusters[kwt_cluster.key] = kwt_cluster
    return full_clusters


def graph_cluster(cluster):
    data = []
    weight = []
    for topic, w in cluster.features.items():
        data.append(int(topic))
        weight.append(int(w))

    # create a horizontal plot
    ax1 = fig.add_subplot(111)
    ax1.eventplot(data)
    plt.show()


def save_clusters(clusters):
    with open(os.path.join(output_folder, 'kwt-clusters-lvl_I.pkl'), 'wb') as out:
        pickle.dump(clusters, out)


def main():
    mappings = load_mappings()
    file_features = load_file_features()
    clusters = generate_clusters(mappings, file_features)
    save_clusters(clusters)
    report_final(clusters)
    # graph_cluster(clusters['KWT-15'])

    return 0


if __name__ == '__main__':
    sys.exit(main())