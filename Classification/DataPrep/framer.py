import datetime
import logging
import os
import pprint
import sys
import timing
import pickle

import pandas as pd
import numpy as np

from DataPrep import config, logger
from lxml import etree


def get_date_from_folder(folder):
    """Extract date from folder name.

    Assumes folder ends with xxx/year/month/day

    :param folder:
    :return: date (YYYY-MM-DD)
    """
    folder_elements = folder.split('/')
    day = int(folder_elements[-1])
    month = int(folder_elements[-2])
    year = int(folder_elements[-3])
    return datetime.date(year, month, day)


def strip_extension(file):
    return file.split('.')[0]


def strip_uri(uri):
    """Extract "raw" topic identifier from uri

    :param uri: "http://kim.oecd.org/Taxonomy/Topics#T187"
    :return: 187
    """
    parts = uri.split("/")
    return parts[-1][8::]


class Extractor:
    CORPUS_ROOT_FOLDER_DEFAULT = ""
    OUTPUT_FOLDER_DEFAULT = "/home/stephane/Playground/PycharmProjects/machine-learning/Classification/DataPrep/output"

    def __init__(self, corpus_root_folder, output_folder, train=False):
        self.logger = logging.getLogger(__name__)

        self.train = train
        self.processed_files = {}
        self.files_features = {}
        self.dated_features = {}
        self.topics = {}
        self.max_topic = 6666

        if corpus_root_folder is not None:
            self.corpus_root_folder = corpus_root_folder
        else:
            self.corpus_root_folder = Extractor.CORPUS_ROOT_FOLDER_DEFAULT

        if output_folder is not None:
            self.output_folder = output_folder
        else:
            self.output_folder = Extractor.OUTPUT_FOLDER_DEFAULT
        if self.train:
            self.output_folder += '/train'

        logger.info("Initialized with corpus root folder: %s", self.corpus_root_folder)
        self.training_jts = self.load_training_jt()
        pprint.pprint(self.training_jts)

    def load_training_jt(self):
        logger.info("Loading training jt")
        with open(os.path.join(self.output_folder, 'mappings.pkl'), 'rb') as input_file:
            mappings = pickle.load(input_file)

        return [jt for jt in mappings.keys()]

    def get_max_topic(self):
        self.max_topic = 0
        idx = 0
        for root, dirs, files_list in os.walk(self.corpus_root_folder):
            logger.debug('root: {r}'.format(r=root))
            logger.debug('dirs: {d}'.format(d=dirs))
            # if idx > 5000:
            #     break
            if not len(files_list) > 0:
                continue
            for semantic_result_file in files_list:
                if os.path.isfile(os.path.join(root, semantic_result_file)):
                    _, file_features = self._process_xml(root, semantic_result_file)
                    idx += 1
                    if idx % 100 == 0:
                        logger.info("%s files processed...", idx)
                    for feature in file_features:
                        if feature > self.max_topic:
                            self.max_topic = feature
                            logger.info('Found new max topic: {m}'.format(m=feature))

        logger.info('Final max topic found: {m}'.format(m=self.max_topic))

    def process_enrichment_files(self):
        idx = 0
        for root, dirs, files_list in os.walk(self.corpus_root_folder):
            logger.debug('root: {r}'.format(r=root))
            logger.debug('dirs: {d}'.format(d=dirs))
            if idx > 50000:
                break
            if not len(files_list) > 0:
                continue
            date = get_date_from_folder(root)
            logger.debug('date: {d}'.format(d=date))
            if date not in self.dated_features:
                self.dated_features[date] = []
            for semantic_result_file in files_list:
                if self.train and strip_extension(semantic_result_file) not in self.training_jts:
                    continue
                if os.path.isfile(os.path.join(root, semantic_result_file)):
                    logger.info("processing: %s/%s", root, semantic_result_file)
                    jt, file_features = self._process_xml(root, semantic_result_file)
                    if jt and file_features:
                        self.files_features[jt] = tuple(file_features)
                        self.dated_features[date].extend(file_features)
                    idx += 1
                    if idx % 100 == 0:
                        logger.info("%s files processed...", idx)

        self.logger.info("Total nb files processed: %s", idx)
        self.create_dataframes()
        self.create_dated_dataframe()
        self.store_vectors()

    def store_vectors(self):
        with open(os.path.join(self.output_folder, 'files_features.pkl'), 'wb') as out:
            pickle.dump(self.files_features, out)
        with open(os.path.join(self.output_folder, 'topics.pkl'), 'wb') as out:
            pickle.dump(self.topics, out)

    def create_dated_dataframe(self):
        # TODO figure out why the resulting dataframe is int64 and not int8
        topics = np.array([int(topic) for topic in self.topics])
        dic = dict()
        for date, features in self.dated_features.items():
            feature_vector = np.zeros(topics.max(0)+1, np.int8)
            for i in features:
                feature_vector[int(i)] += 1
            dic[date] = feature_vector
        df = pd.DataFrame.from_dict(dic, orient='index')
        logger.info("Dated DataFrame created")

        # Store to file system using hdf format
        self.store_as_hdf(df=df, target_file_name='dated_feature_vectors.hdf', key='topics_usage')
        logger.info("Dated DataFrame saved to disk")

    def create_dataframes(self):
        topics = np.array([int(topic) for topic in self.topics])
        logger.debug("topics max: {max}".format(max=topics.max(0)))
        # logger.debug("self.files_features.keys(): {keys}".format(keys=self.files_features.keys()))

        files_array = np.array([key for key in self.files_features.keys()])
        logger.debug("files_array.size: {size}".format(size=files_array.size))

        vectors_list = []

        timing.start_clock("Start Feature_Vectors creation")
        for jt, features in self.files_features.items():
            feature_vector = np.zeros(topics.max(0)+1, np.int8)
            for i in features:
                feature_vector[int(i)] = 1
            vectors_list.append(feature_vector)

        logger.debug("len vectors_list: {len}".format(len=len(vectors_list)))
        feature_vectors = np.concatenate(vectors_list, axis=0)
        logger.debug("feature vectors concatenated")
        feature_vectors = feature_vectors.reshape((files_array.size, topics.max(0)+1))
        logger.debug("feature vectors reshaped")

        df = pd.DataFrame(feature_vectors, index=files_array)

        self.store_as_hdf(df=df, target_file_name='feature_vectors.hdf', key='feature_vectors')
        timing.stop_clock()
        logger.info("Feature_Vectors created successfully")

        return

    def _process_xml(self, folder, result_file):
        # First of all, take care of potential duplicates in the corpus
        jt = strip_extension(result_file)
        if result_file in self.processed_files:
            logger.info("File %s already processed.", result_file)
            if not self._is_posterior(result_file, folder):
                logger.info("A more recent version of %s was already processed, moving on", result_file)
                return
            else:
                # remove previous occurrences in various indexes to be replaced by this one
                logger.info("An older version of %s was processed, replacing with most recent one.", result_file)
                self.files_features.pop(jt)

        try:
            doc = etree.parse(os.path.join(folder, result_file))
        except Exception as e:
            logger.error("Failed to load xml content for file: %s", result_file, exc_info=True)
            return jt, []

        root = doc.getroot()
        file_features = []

        for subject in root.findall("./annotation/subject"):
            topic_id = int(strip_uri(subject.get('uri')))
            label_en = subject.get('label_en')
            # relevance = 'N' if subject.get('relevance') == 'normal' else 'H'
            file_features.append(topic_id)
            if topic_id not in self.topics:
                self.topics[topic_id] = label_en

        self.processed_files[result_file] = folder
        return jt, file_features

    def _is_posterior(self, result_file, current_folder):
        previous_folder = self.processed_files[result_file]

        previous_date = get_date_from_folder(previous_folder)
        current_date = get_date_from_folder(current_folder)

        if previous_date < current_date:
            return True
        else:
            return False

    def store_as_hdf(self, df, target_file_name, key):
        """
        Store dataframe to file system using hdf format
        :param df:
        :param target_file_name:
        :param key:
        :return: None
        """
        df.to_hdf(os.path.join(self.output_folder, target_file_name), key=key)
        logger.info("Dated DataFrame saved to disk")
        logger.debug(df.head(10))
        df.info()


def main():

    corpus_root = config.get('MAIN', 'corpus_root')
    output_dir = config.get('MAIN', 'output_dir')

    extractor = Extractor(corpus_root_folder=corpus_root, output_folder=output_dir, train=True)
    extractor.process_enrichment_files()

    logger.info("File processing complete.")
    # logger.debug("Files features: {features}".format(features=extractor.files_features))
    logger.info("Nb of files: {nb}".format(nb=len(extractor.files_features)))
    # logger.debug("Topics: {topics}".format(topics=extractor.topics))
    logger.info("Nb of topics: {nb}".format(nb=len(extractor.topics)))

if __name__ == '__main__':
    sys.exit(main())
