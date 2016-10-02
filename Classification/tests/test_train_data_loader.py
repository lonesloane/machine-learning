import unittest
from DataPrep import train_data_loader


class TrainDataLoaderTestCase(unittest.TestCase):

    def test_extract_indices(self):
        kwt = 'KWT-1-1'
        main, second = train_data_loader.extract_indices(kwt)
        self.assertEqual(1, main)
        self.assertNotEquals(2, second)
        self.assertEquals(1, second)

        kwt = 'KWT-2-12'
        main, second = train_data_loader.extract_indices(kwt)
        self.assertEqual(2, main)
        self.assertEquals(12, second)

    def test_is_lower(self):
        kwt1 = 'KWT-1-2'
        kwt2 = 'KWT-2-1'
        self.assertTrue(train_data_loader.is_lower(kwt1, kwt2))
        kwt1 = 'KWT-3-2'
        kwt2 = 'KWT-2-1'
        self.assertFalse(train_data_loader.is_lower(kwt1, kwt2))
        kwt1 = 'KWT-1-2'
        kwt2 = 'KWT-1-1'
        self.assertFalse(train_data_loader.is_lower(kwt1, kwt2))
        kwt1 = 'KWT-1-1'
        kwt2 = 'KWT-1-2'
        self.assertTrue(train_data_loader.is_lower(kwt1, kwt2))


if __name__ == '__main__':
    unittest.main()
