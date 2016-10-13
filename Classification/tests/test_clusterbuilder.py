import unittest
import DataPrep.clusterbuilder as cbuilder


class ClusterBuilderTestCase(unittest.TestCase):

    def test_init(self):
        builder = cbuilder.Builder()
        self.assertEqual(False, builder._train)

        builder = cbuilder.Builder(train=True)
        self.assertEqual(True, builder._train)

    def test_load_file_features(self):
        builder = cbuilder.Builder()

        self.assertDictEqual({}, builder.file_features)
        builder.load_file_features()
        self.assertTrue(len(builder.file_features.keys()) > 0)

    def test_get_proximity(self):
        v1 = {1, 3, 5}
        v2 = {1, 2, 3, 4, 5, 6}
        proximity = cbuilder.get_proximity(v1, v2)
        self.assertEqual(3, proximity)

        v1 = {0, 7, 9}
        v2 = {1, 2, 3, 4, 5, 6}
        proximity = cbuilder.get_proximity(v1, v2)
        self.assertEqual(0, proximity)

        v1 = {1, 2, 3, 4, 5, 6}
        v2 = {1, 2, 3, 4, 5, 6}
        proximity = cbuilder.get_proximity(v1, v2)
        self.assertEqual(6, proximity)

    @unittest.skip(reason='Not implemented yet.')
    def test_add_cluster(self):
        pass

    @unittest.skip(reason='Not implemented yet.')
    def test_clear_cluster(self):
        pass


if __name__ == '__main__':
    unittest.main()
