import re
import unittest
import uuid

import DataPrep.clusterbuilder as builder


def sorted_tuples(k1, k2, w):
    if k1 < k2:
        return (k1, k2, w)
    else:
        return (k2, k1, w)


class ClusterBuilderTestCase(unittest.TestCase):

    def test_get_proximity(self):
        f1 = {1: 1, 5: 1}
        f2 = {1: 1, 5: 1}
        proximity = builder.get_proximity(f1, f2)
        self.assertEqual(1.0, proximity)

        f1 = {1: 1, 3: 1, 5: 1}
        f2 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        proximity = builder.get_proximity(f1, f2)
        self.assertEqual(3/6, proximity)

        f1 = {1: 10, 3: 10, 5: 1}
        f2 = {1: 1, 2: 1, 3: 10, 4: 1, 5: 1, 6: 1}
        proximity = builder.get_proximity(f1, f2)
        self.assertEqual(111/6, proximity)

        f1 = {0: 1, 7: 1, 9: 1}
        f2 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        proximity = builder.get_proximity(f1, f2)
        self.assertEqual(0, proximity)

        f1 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        f2 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        proximity = builder.get_proximity(f1, f2)
        self.assertEqual(1.0, proximity)

    def test_get_cluster_proximity(self):
        component1 = {'jt01': {1: 1, 3: 1, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 1, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        components1 = {cluster1.key: cluster1,
                       cluster2.key: cluster2}
        cluster10 = builder.Cluster(components1, root=False)
        component3 = {'jt03': {1: 1, 3: 1, 4: 1}}
        cluster3 = builder.Cluster(component3, root=True)
        component4 = {'jt04': {1: 1, 2: 1, 5: 1}}
        cluster4 = builder.Cluster(component4, root=True)
        components2 = {cluster3.key: cluster3,
                       cluster4.key: cluster4}
        cluster20 = builder.Cluster(components2, root=False)
        proximity = builder.get_clusters_proximity(cluster1=cluster10,
                                                   cluster2=cluster20)
        self.assertAlmostEqual(0.55, proximity, 6)

        component1 = {'jt11': {88: 10, 86: 10, 6: 10, 39: 100, 24: 100, 72: 10, 41: 100, 62: 10, 76: 10, 14: 10, 53: 10}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt08': {48: 100, 32: 100, 70: 100, 56: 100, 73: 100, 42: 10, 27: 10, 76: 100, 29: 100, 78: 100, 47: 10}}
        cluster2 = builder.Cluster(component2, root=True)
        proximity = builder.get_clusters_proximity(cluster1=cluster1,
                                                   cluster2=cluster2)
        self.assertAlmostEqual(47.62, proximity, 2)

    def test_get_clusters_common_keys(self):
        component1 = {'jt01': {1: 1, 3: 1, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 1, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        components1 = {cluster1.key: cluster1,
                       cluster2.key: cluster2}
        cluster10 = builder.Cluster(components1, root=False)
        component3 = {'jt03': {1: 1, 3: 1, 4: 1}}
        cluster3 = builder.Cluster(component3, root=True)
        component4 = {'jt02': {1: 1, 2: 1, 5: 1}}
        cluster4 = builder.Cluster(component4, root=True)
        components2 = {cluster3.key: cluster3,
                       cluster4.key: cluster4}
        cluster20 = builder.Cluster(components2, root=False)
        common_keys = builder.get_clusters_common_keys(cluster1=cluster10,
                                                       cluster2=cluster20)
        self.assertListEqual(list(common_keys), ['jt02'])

        component1 = {'jt01': {1: 1, 3: 1, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 1, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        component5 = {'jt05': {1: 1, 2: 1, 5: 1}}
        cluster5 = builder.Cluster(component5, root=True)
        components1 = {cluster1.key: cluster1,
                       cluster2.key: cluster2,
                       cluster5.key: cluster5}
        cluster10 = builder.Cluster(components1, root=False)
        component3 = {'jt03': {1: 1, 3: 1, 4: 1}}
        cluster3 = builder.Cluster(component3, root=True)
        component4 = {'jt02': {1: 1, 2: 1, 5: 1}}
        cluster4 = builder.Cluster(component4, root=True)
        component6 = {'jt05': {1: 1, 2: 1, 5: 1}}
        cluster6 = builder.Cluster(component6, root=True)
        components2 = {cluster3.key: cluster3,
                       cluster4.key: cluster4,
                       cluster6.key: cluster6}
        cluster20 = builder.Cluster(components2, root=False)
        common_keys = builder.get_clusters_common_keys(cluster1=cluster10,
                                                       cluster2=cluster20)
        self.assertListEqual(sorted(list(common_keys)), ['jt02', 'jt05'])

    def test_trim_cluster(self):
        clusters = {'jt01': [('jt03', 2), ('jt04', 5)],
                    'jt02': [('jt05', 4), ('jt06', 1)]}
        builder.trim_cluster(clusters, key='jt01', component=('jt04', 5), strict=True)
        self.assertEqual([('jt03', 2)], clusters['jt01'])
        builder.trim_cluster(clusters, key='jt01', component=('jt03', 2), strict=True)
        self.assertDictEqual({'jt02': [('jt05', 4), ('jt06', 1)]}, clusters)

        clusters = {'jt01': [('jt03', 2), ('jt04', 5)],
                    'jt02': [('jt05', 4), ('jt06', 1)]}
        builder.trim_cluster(clusters, key='jt01', component=('jt04', 5), strict=False)
        self.assertEqual([('jt03', 2)], clusters['jt01'])
        builder.trim_cluster(clusters, key='jt01', component=('jt03', 2), strict=False)
        self.assertDictEqual({'jt01':[], 'jt02': [('jt05', 4), ('jt06', 1)]}, clusters)


class BuilderTestCase(unittest.TestCase):

    def setUp(self):
        builder.cluster_key_index = 0

        o_builder = builder.Builder(input_data_file='Dummy.csv')

        cluster = builder.Cluster({'jt01': {1: 1, 2: 1, 3: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt02': {1: 1, 2: 1, 5: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt03': {7: 1, 2: 1, 6: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt04': {6: 1, 3: 1, 7: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt05': {1: 1, 2: 1, 3: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster

        component1 = {'jt01': {1: 2, 3: 2, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 5, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        component3 = {'jt03': {1: 1, 3: 1, 4: 1}}
        cluster3 = builder.Cluster(component3, root=True)
        component4 = {'jt04': {1: 1, 2: 1, 4: 1}}
        cluster4 = builder.Cluster(component4, root=True)
        component5 = {'jt05': {1: 1, 2: 1, 5: 1}}
        cluster5 = builder.Cluster(component5, root=True)
        components1 = {cluster1.key: cluster1,
                       cluster2.key: cluster2,
                       cluster3.key: cluster3,
                       cluster4.key: cluster4,
                       cluster5.key: cluster5}
        cluster10 = builder.Cluster(components1, root=False)
        o_builder.full_clusters_list[cluster10.key] = cluster10
        o_builder.current_clusters[cluster10.key] = cluster10

        component6 = {'jt06': {1: 2, 3: 2, 5: 1}}
        cluster6 = builder.Cluster(component6, root=True)
        component7 = {'jt07': {1: 1, 2: 5, 5: 1}}
        cluster7 = builder.Cluster(component7, root=True)
        component8 = {'jt08': {1: 1, 3: 1, 4: 1}}
        cluster8 = builder.Cluster(component8, root=True)
        component9 = {'jt09': {1: 1, 2: 1, 4: 1}}
        cluster9 = builder.Cluster(component9, root=True)
        components2 = {cluster6.key: cluster6,
                       cluster7.key: cluster7,
                       cluster8.key: cluster8,
                       cluster9.key: cluster9}
        cluster20 = builder.Cluster(components2, root=False)
        o_builder.full_clusters_list[cluster20.key] = cluster20
        o_builder.current_clusters[cluster20.key] = cluster20

        self.builder = o_builder

    def test_init(self):
        o_builder = builder.Builder(input_data_file='dummy.csv', graph_format='format')
        self.assertEqual('dummy.csv', o_builder.input_data_file)
        self.assertEqual('format', o_builder.graph_format)
        self.assertDictEqual({}, o_builder.full_clusters_list)
        self.assertDictEqual({}, o_builder.current_clusters)
        self.assertDictEqual({}, o_builder.bound_clusters)
        self.assertEqual(0, o_builder.unique_id)

    def test_load_individual_components(self):
        o_builder = builder.Builder('test_files_features.pkl')
        o_builder.output_folder = '/home/stephane/Playground/PycharmProjects/machine-learning/Classification/tests/data/'

        self.assertDictEqual({}, o_builder.current_clusters)
        o_builder.load_individual_components()
        self.assertEquals(len(o_builder.current_clusters.keys()),  5)
        self.assertEquals(len(o_builder.full_clusters_list.keys()),  5)
        '''
        self.assertEquals(len(builder.atoms.keys()),  5)
        self.assertTrue('jt01' in builder.atoms.keys())
        self.assertTrue('jt02' in builder.atoms.keys())
        self.assertTrue('jt03' in builder.atoms.keys())
        self.assertTrue('jt04' in builder.atoms.keys())
        self.assertTrue('jt05' in builder.atoms.keys())
        self.assertDictEqual({1: 1, 2: 1, 3: 1},
                             builder.atoms['jt01'].features)
        self.assertDictEqual({1: 1, 2: 1, 5: 1},
                             builder.atoms['jt02'].features)
        self.assertDictEqual({7: 1, 2: 1, 6: 1},
                             builder.atoms['jt03'].features)
        self.assertDictEqual({6: 1, 3: 1, 7: 1},
                             builder.atoms['jt04'].features)
        self.assertDictEqual({1: 1, 2: 1, 3: 1},
                             builder.atoms['jt05'].features)
        '''
        self.assertTrue('jt01' in o_builder.current_clusters.keys())
        self.assertTrue('jt02' in o_builder.current_clusters.keys())
        self.assertTrue('jt03' in o_builder.current_clusters.keys())
        self.assertTrue('jt04' in o_builder.current_clusters.keys())
        self.assertTrue('jt05' in o_builder.current_clusters.keys())
        self.assertDictEqual({1: 1, 2: 1, 3: 1},
                             o_builder.current_clusters['jt01'].features)
        self.assertDictEqual({1: 1, 2: 1, 5: 1},
                             o_builder.current_clusters['jt02'].features)
        self.assertDictEqual({7: 1, 2: 1, 6: 1},
                             o_builder.current_clusters['jt03'].features)
        self.assertDictEqual({6: 1, 3: 1, 7: 1},
                             o_builder.current_clusters['jt04'].features)
        self.assertDictEqual({1: 1, 2: 1, 3: 1},
                             o_builder.current_clusters['jt05'].features)

    def test_clear_bound_clusters(self):
        o_builder = builder.Builder(input_data_file='dummy.csv')
        o_builder.bound_clusters = dict()
        o_builder.bound_clusters['jt01'] = [1, 2, 3]
        o_builder.bound_clusters['jt02'] = [3, 2, 1]
        self.assertEqual([1,2,3], o_builder.bound_clusters['jt01'])
        self.assertEqual([3,2,1], o_builder.bound_clusters['jt02'])
        o_builder.clear_bound_clusters('jt02')
        self.assertFalse('jt02' in o_builder.bound_clusters)
        o_builder.clear_bound_clusters('jt01')
        self.assertFalse('jt01' in o_builder.bound_clusters)

    def test_merge_clusters(self):
        self.builder.bound_clusters = {'cls_1': [('cls_2', 1)]}
        self.builder.merge_clusters()
        self.assertEqual(1, len(self.builder.current_clusters))
        self.assertTrue('cls_3' in self.builder.current_clusters.keys())
        for cluster_key in self.builder.current_clusters:
            print('cluster: %s - length: %s - mean proximity: %s - features: %s' %
                  (cluster_key,
                   len(self.builder.full_clusters_list[cluster_key].components),
                   self.builder.full_clusters_list[cluster_key].mean_proximity,
                   self.builder.full_clusters_list[cluster_key].features))
            print('inner components: %s ' % sorted(self.builder.full_clusters_list[cluster_key].components.keys()))

        self.builder.bound_clusters = {'jt02': [('jt05', 2.75),
                                                ('jt01', 2.75),
                                                ('jt03', 0.20000000000000001)],
                                       'jt05': [('jt04', 2.0),
                                                ('jt01', 10.0),
                                                ('jt03', 0.20000000000000001),
                                                ('jt02', 2.75)],
                                       'jt04': [('jt05', 2.0),
                                                ('jt01', 0.20000000000000001),
                                                ('jt03', 2.75)],
                                       'jt03': [('jt04', 2.75),
                                                ('jt05', 0.20000000000000001),
                                                ('jt01', 2.0),
                                                ('jt02', 0.20000000000000001)],
                                       'jt01': [('jt04', 0.20000000000000001),
                                                ('jt05', 10.0),
                                                ('jt03', 2.0),
                                                ('jt02', 2.75)]}
        self.builder.merge_clusters()
        self.assertEqual(5, len(self.builder.current_clusters))
        generated_clusters_keys = [tuple(sorted(self.builder.full_clusters_list[cluster_key].components.keys()))
                                   for cluster_key in self.builder.current_clusters]
        #self.assertTrue(('jt02',) in generated_clusters_keys)
        #self.assertTrue(('jt01', 'jt05') in generated_clusters_keys)
        #self.assertTrue(('jt03', 'jt04') in generated_clusters_keys)

        print(generated_clusters_keys)
        for cluster_key in self.builder.current_clusters:
            print('cluster: %s - length: %s - mean proximity: %s - features: %s' %
                  (cluster_key,
                   len(self.builder.full_clusters_list[cluster_key].components),
                   self.builder.full_clusters_list[cluster_key].mean_proximity,
                   self.builder.full_clusters_list[cluster_key].features))
            print('inner components: %s ' % sorted(self.builder.full_clusters_list[cluster_key].components.keys()))

    def test_remove_duplicates(self):
        self.builder.bound_clusters = {'jt02': [('jt05', 2.75),
                                                ('jt01', 2.75),
                                                ('jt03', 0.20000000000000001)],
                                       'jt05': [('jt04', 2.0),
                                                ('jt01', 10.0),
                                                ('jt03', 0.20000000000000001),
                                                ('jt02', 2.75)],
                                       'jt04': [('jt05', 2.0),
                                                ('jt01', 0.20000000000000001),
                                                ('jt03', 2.75)],
                                       'jt03': [('jt04', 2.75),
                                                ('jt05', 0.20000000000000001),
                                                ('jt01', 2.0),
                                                ('jt02', 0.20000000000000001)],
                                       'jt01': [('jt04', 0.20000000000000001),
                                                ('jt05', 10.0),
                                                ('jt03', 2.0),
                                                ('jt02', 2.75)]}
        self.builder.merge_clusters()
        cluster_components = []
        for cluster_key, cluster in self.builder.current_clusters.items():
            cluster_components.append(tuple(sorted(cluster.components.keys())))
        print(cluster_components)
        self.assertEqual(3, cluster_components.count(('jt01', 'jt02', 'jt03', 'jt04', 'jt05')))
        self.assertEqual(1, cluster_components.count(('jt01', 'jt02', 'jt03', 'jt05')))
        self.assertEqual(1, cluster_components.count(('jt01', 'jt03', 'jt04', 'jt05')))
        self.builder.remove_duplicates()
        cluster_components = []
        for cluster_key, cluster in self.builder.current_clusters.items():
            cluster_components.append(tuple(sorted(cluster.components.keys())))
        print(cluster_components)
        self.assertEqual(1, cluster_components.count(('jt01', 'jt02', 'jt03', 'jt04', 'jt05')))
        self.assertEqual(1, cluster_components.count(('jt01', 'jt02', 'jt03', 'jt05')))
        self.assertEqual(1, cluster_components.count(('jt01', 'jt03', 'jt04', 'jt05')))

    def test_nb_of_components(self):
        o_builder = builder.Builder('dummy.csv')

        component1 = {'jt01': {1: 2, 3: 2, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 5, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        component3 = {'jt03': {1: 1, 3: 1, 4: 1}}
        cluster3 = builder.Cluster(component3, root=True)
        component4 = {'jt04': {1: 1, 2: 1, 4: 1}}
        cluster4 = builder.Cluster(component4, root=True)
        component5 = {'jt05': {1: 1, 2: 1, 5: 1}}
        cluster5 = builder.Cluster(component5, root=True)
        components = {cluster1.key: cluster1,
                      cluster2.key: cluster2,
                      cluster3.key: cluster3,
                      cluster4.key: cluster4,
                      cluster5.key: cluster5}
        cluster = builder.Cluster(components, root=False)
        o_builder.current_clusters[cluster.key] = cluster

        self.assertEqual(5, o_builder.nb_of_atoms())

        o_builder.current_clusters = {}
        component1 = {'jt01': {1: 2, 3: 2, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 5, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        components = {cluster1.key: cluster1,
                      cluster2.key: cluster2}
        cluster = builder.Cluster(components, root=False)
        o_builder.current_clusters[cluster.key] = cluster
        self.assertEqual(2, o_builder.nb_of_atoms())

        component3 = {'jt03': {1: 1, 3: 1, 4: 1}}
        cluster3 = builder.Cluster(component3, root=True)
        component4 = {'jt04': {1: 1, 2: 1, 4: 1}}
        cluster4 = builder.Cluster(component4, root=True)
        components = {cluster3.key: cluster3,
                      cluster4.key: cluster4}
        cluster = builder.Cluster(components, root=False)
        o_builder.current_clusters[cluster.key] = cluster
        self.assertEqual(4, o_builder.nb_of_atoms())

        component1 = {'jt05': {1: 2, 3: 2, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt06': {1: 1, 2: 5, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        component5 = {'jt02': {1: 1, 2: 5, 5: 1}}
        cluster5 = builder.Cluster(component5, root=True)
        components = {cluster1.key: cluster1,
                      cluster2.key: cluster2,
                      cluster5.key: cluster5}
        cluster = builder.Cluster(components, root=False)
        o_builder.current_clusters[cluster.key] = cluster
        self.assertEqual(6, o_builder.nb_of_atoms())


class BuilderDipolesTestCase(unittest.TestCase):

    def setUp(self):
        o_builder = builder.BuilderDiPoles(input_data_file='Dummy.csv')
        cluster = builder.Cluster({'jt01': {1: 1, 2: 10, 3: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt02': {1: 1, 2: 1, 5: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt03': {7: 10, 2: 1, 6: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt04': {6: 1, 3: 1, 7: 1}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt05': {1: 10, 2: 1, 3: 10}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        cluster = builder.Cluster({'jt07': {1: 10, 2: 1, 3: 10}}, root=True)
        o_builder.full_clusters_list[cluster.key] = cluster
        o_builder.current_clusters[cluster.key] = cluster
        self.builder = o_builder

    def test_generate_bounds(self):
        self.builder.generate_bounds()
        clusters = self.builder.bound_clusters
        print('clusters:\n%s' % clusters)
        self.assertTrue('jt01' in clusters.keys())
        self.assertListEqual(sorted([('jt02', 2.75), ('jt03', 2.0),
                                     ('jt04', 0.20000000000000001),
                                     ('jt05', 10.0),
                                     ('jt07', 10.0)]),
                             sorted(clusters['jt01']))
        self.assertTrue('jt02' in clusters.keys())
        self.assertListEqual(sorted([('jt01', 2.75),
                                     ('jt03', 0.20000000000000001),
                                     ('jt05', 2.75),
                                     ('jt07', 2.75)]),
                             sorted(clusters['jt02']))
        self.assertTrue('jt03' in clusters.keys())
        self.assertEqual(sorted([('jt01', 2.0),
                                ('jt02', 0.20000000000000001),
                                ('jt04', 2.75),
                                ('jt05', 0.20000000000000001),
                                ('jt07', 0.20000000000000001)]),
                         sorted(clusters['jt03']))
        self.assertTrue('jt04' in clusters.keys())
        self.assertEqual(sorted([('jt01', 0.20000000000000001),
                                 ('jt03', 2.75),
                                 ('jt05', 2.0),
                                 ('jt07', 2.0)]),
                         sorted(clusters['jt04']))
        self.assertTrue('jt05' in clusters.keys())
        self.assertEqual(sorted([('jt01', 10.0),
                                 ('jt02', 2.75),
                                 ('jt03', 0.20000000000000001),
                                 ('jt04', 2.0),
                                 ('jt07', 67.0)]),
                         sorted(clusters['jt05']))

    def test_trim_bounds(self):
        self.fail('Not implemented')

    def test_add_bound_cluster(self):
        self.assertDictEqual({}, self.builder.bound_clusters)
        self.builder.add_bound_cluster('jt01', 'jt02', 2)
        self.assertDictEqual({'jt01': [('jt02', 2)]}, self.builder.bound_clusters)
        self.builder.add_bound_cluster('jt01', 'jt03', 4)
        self.assertDictEqual({'jt01': [('jt02', 2),
                                       ('jt03', 4)]}, self.builder.bound_clusters)
        self.builder.add_bound_cluster('jt02', 'jt04', 2)
        self.assertDictEqual({'jt02': [('jt04', 2)],
                              'jt01': [('jt02', 2), ('jt03', 4)]}
                             , self.builder.bound_clusters)

    def test_trim_clusters(self):
        self.builder.bound_clusters = {'jt02': [('jt05', 2.75),
                                                ('jt01', 2.75),
                                                ('jt03', 0.20000000000000001)],
                                       'jt05': [('jt04', 2.0),
                                                ('jt01', 10.0),
                                                ('jt03', 0.20000000000000001),
                                                ('jt02', 2.75)],
                                       'jt04': [('jt05', 2.0),
                                                ('jt01', 0.20000000000000001),
                                                ('jt03', 2.75)],
                                       'jt03': [('jt04', 2.75),
                                                ('jt05', 0.20000000000000001),
                                                ('jt01', 2.0),
                                                ('jt02', 0.20000000000000001)],
                                       'jt01': [('jt04', 0.20000000000000001),
                                                ('jt05', 10.0),
                                                ('jt03', 2.0),
                                                ('jt02', 2.75)]}
        self.builder.merge_clusters()
        print('='*30)
        print('mean clusters proximity: {mp}'.format(mp=self.builder.get_clusters_mean_proximity()))
        for key, cluster in self.builder.current_clusters.items():
            print(key)
            print(cluster.components)
            print('mean proximity: {mp}'.format(mp=cluster.mean_proximity))
        print('='*30)
        self.builder.trim_clusters()
        for key, cluster in self.builder.current_clusters.items():
            print(key)
            print(cluster.components)
            print('mean proximity: {mp}'.format(mp=cluster.mean_proximity))
            print('=' * 30)

    def test_get_clusters_mean_proximity(self):
        self.builder.bound_clusters = {'jt01': [('jt05', 3)],
                                       'jt03': [('jt04', 2)],
                                       'jt04': [('jt03', 2)],
                                       'jt02': [],
                                       'jt05': [('jt01', 3)]}
        self.builder.merge_clusters()
        self.assertAlmostEqual(2.55, self.builder.get_clusters_mean_proximity(), 4)
        self.builder.bound_clusters = {'jt01': [('jt05', 3), ('jt07', 1)],
                                       'jt03': [('jt04', 2)],
                                       'jt04': [('jt03', 2)],
                                       'jt02': [],
                                       'jt05': [('jt01', 4)]}
        self.builder.merge_clusters()
        self.assertAlmostEqual(7.35, self.builder.get_clusters_mean_proximity(), 4)

    def test_trim_clusters_mean_proximity(self):
        self.fail('Not implemented')

    def test_remove_double_edges(self):
        self.assertDictEqual({}, self.builder.bound_clusters)
        self.builder.bound_clusters = {'jt01': [('jt05', 3)],
                                       'jt03': [('jt04', 2)],
                                       'jt04': [('jt03', 2)],
                                       'jt02': [],
                                       'jt05': [('jt01', 3)]}
        self.builder.remove_double_edges()
        print(self.builder.bound_clusters)
        self.assertTrue('jt01' in self.builder.bound_clusters.keys())
        self.assertTrue('jt02' in self.builder.bound_clusters.keys())
        self.assertTrue('jt03' in self.builder.bound_clusters.keys())
        self.assertEqual([('jt05', 3)], self.builder.bound_clusters['jt01'])
        self.assertEqual([], self.builder.bound_clusters['jt02'])
        self.assertEqual([('jt04', 2)], self.builder.bound_clusters['jt03'])


class ClusterTestCase(unittest.TestCase):

    def test_init(self):

        # TODO: initial feature weights are lost. Should it be the case?
        components = {'jt01': {1: 10, 3: 10, 5: 10}}
        cluster = builder.Cluster(components, root=True)
        self.assertEqual(components, cluster.components)
        self.assertEqual(1, cluster.length)
        self.assertEqual(0, cluster.mean_proximity)
        self.assertDictEqual({1: 10,
                              3: 10,
                              5: 10}, cluster.features)
        self.assertDictEqual({'jt01': {1: 10, 3: 10, 5: 10}},
                             cluster.atoms)

        components = {'jt01': {1: 1, 3: 1, 5: 1}}
        cluster = builder.Cluster(components, root=True)
        self.assertEqual(components, cluster.components)
        self.assertEqual(1, cluster.length)
        self.assertEqual(0, cluster.mean_proximity)
        self.assertDictEqual({1: 1,
                              3: 1,
                              5: 1}, cluster.features)
        self.assertDictEqual({'jt01': {1: 1, 3: 1, 5: 1}},
                             cluster.atoms)

        component1 = {'jt01': {1: 1, 3: 1, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 1, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        components = {cluster1.key: cluster1,
                      cluster2.key: cluster2}
        cluster = builder.Cluster(components, root=False)
        self.assertEqual(components, cluster.components)
        self.assertEqual(2, cluster.length)
        self.assertEqual(0.25, cluster.mean_proximity)
        self.assertDictEqual({1: 2,
                              2: 1,
                              3: 1,
                              5: 2}, cluster.features)
        self.assertDictEqual({'jt01': {1: 1, 3: 1, 5: 1},
                              'jt02': {1: 1, 2: 1, 5: 1}},
                             cluster.atoms)

        components = {'jt01': {1: 10, 3: 10, 5: 10}}
        cluster = builder.Cluster(components, root=True)
        self.assertEqual(components, cluster.components)
        self.assertEqual(1, cluster.length)
        self.assertEqual(0, cluster.mean_proximity)
        self.assertDictEqual({1: 10,
                              3: 10,
                              5: 10}, cluster.features)

        component1 = {'jt01': {1: 2, 3: 2, 5: 1}}
        cluster1 = builder.Cluster(component1, root=True)
        component2 = {'jt02': {1: 1, 2: 5, 5: 1}}
        cluster2 = builder.Cluster(component2, root=True)
        components = {cluster1.key: cluster1,
                      cluster2.key: cluster2}
        cluster = builder.Cluster(components, root=False)
        self.assertEqual(components, cluster.components)
        self.assertEqual(2, cluster.length)
        self.assertEqual(0.375, cluster.mean_proximity)
        self.assertDictEqual({1: 3,
                              2: 5,
                              3: 2,
                              5: 2}, cluster.features)

    def test_process_atoms(self):
        atom = {'jt01': {1: 10, 3: 10, 5: 10}}
        cluster1 = builder.Cluster(components=atom, root=True)
        atom = {'jt02': {1: 10, 2: 1, 5: 100}}
        cluster2 = builder.Cluster(components=atom, root=True)
        components = dict()
        components[cluster1.key]=cluster1
        components[cluster2.key]=cluster2

        atoms1 = builder.Cluster.process_atoms(components)
        self.assertDictEqual({'jt01': {1: 10, 3: 10, 5: 10},
                              'jt02': {1: 10, 2: 1, 5: 100}},
                             atoms1)
        cluster10 = builder.Cluster({"dummy":{}}, root=True)
        cluster10.atoms = atoms1

        atom = {'jt01': {1: 10, 3: 10, 5: 10}}
        cluster1 = builder.Cluster(components=atom, root=True)
        atom = {'jt03': {1: 10, 3: 1, 6: 100}}
        cluster2 = builder.Cluster(components=atom, root=True)
        atom = {'jt04': {6: 10, 7: 1, 8: 100}}
        cluster3 = builder.Cluster(components=atom, root=True)
        components = dict()
        components[cluster1.key]=cluster1
        components[cluster2.key]=cluster2
        components[cluster3.key]=cluster3

        atoms2 = builder.Cluster.process_atoms(components)
        self.assertDictEqual({'jt01': {1: 10, 3: 10, 5: 10},
                              'jt03': {1: 10, 3: 1, 6: 100},
                              'jt04': {6: 10, 7: 1, 8: 100}},
                             atoms2)

        cluster20 = builder.Cluster({"dummy": {}}, root=True)
        cluster20.atoms = atoms2
        components = {'cl1': cluster10, 'cl2': cluster20}
        atoms = builder.Cluster.process_atoms(components)
        self.assertDictEqual({'jt03': {1: 10, 3: 1, 6: 100},
                              'jt04': {8: 100, 6: 10, 7: 1},
                              'jt01': {1: 10, 3: 10, 5: 10},
                              'jt02': {1: 10, 2: 1, 5: 100}},
                             atoms)

    def test_process_features(self):
        components = dict()

        components['jt01'] = {1: 10, 3: 10, 5: 10}
        features = builder.Cluster.process_features(components)
        self.assertEqual(3, len(features.keys()))
        self.assertDictEqual({1: 10,
                              3: 10,
                              5: 10}, features)

        components['jt02'] = {1: 10, 2: 1, 5: 100}
        features = builder.Cluster.process_features(components)
        self.assertEqual(4, len(features.keys()))
        self.assertDictEqual({1: 20,
                              2: 1,
                              3: 10,
                              5: 110}, features)

    def test_get_mean_proximity(self):
        components = dict()
        components['jt01'] = {1: 1, 3: 1, 5: 1}
        components['jt02'] = {1: 1, 2: 1, 5: 1}
        mean_proximity = builder.Cluster.get_mean_proximity(components)
        self.assertEqual(2./4./2., mean_proximity)
        components['jt03'] = {1: 1, 3: 1, 4: 1}
        mean_proximity = builder.Cluster.get_mean_proximity(components)
        self.assertEqual(6./5./3., mean_proximity)
        components['jt04'] = {1: 1, 2: 1, 4: 1}
        mean_proximity = builder.Cluster.get_mean_proximity(components)
        self.assertAlmostEqual(12./5./4., mean_proximity, 3)

        components = dict()
        components['jt01'] = {1: 10, 3: 1, 5: 1}
        components['jt02'] = {1: 1, 2: 1, 5: 1}
        mean_proximity = builder.Cluster.get_mean_proximity(components)
        self.assertEqual(11./4./2., mean_proximity)
        components['jt03'] = {1: 1, 3: 10, 4: 1}
        mean_proximity = builder.Cluster.get_mean_proximity(components)
        self.assertEqual(2.65, mean_proximity)

    def test_create_key(self):
        builder.cluster_key_index = 0
        components = {'jt01': {1: 10, 3: 10, 5: 10}}
        key = builder.Cluster.create_key(components, root=True)
        self.assertEqual(key, 'jt01')
        components['JT02'] = {1: 1, 2: 5, 5: 1}
        key = builder.Cluster.create_key(components, root=False)
        self.assertEqual('cls_1', key)
        # ptrn_uuid = re.compile('\w{8}-\w{4}-\w{4}-\w{4}-\w{8}')
        # self.assertTrue(re.search(ptrn_uuid,key))

if __name__ == '__main__':
    unittest.main()
