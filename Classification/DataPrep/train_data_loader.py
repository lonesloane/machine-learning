import json
import os
import pickle
import sys
import pprint
from DataPrep import logger

training_data_folder = '/media/stephane/Storage/OECD/Machine Learning/Mappings/Corpora'
topics_definition_folder = '/media/stephane/Storage/OECD/Machine Learning/Mappings/KappaWebTopics/'
output_folder = 'output/train'


def extract_indices(kwt):
    parts = kwt.split('-')
    return int(parts[1]), int(parts[2])


def is_lower(kwt1, kwt2):
    main_1, second_1 = extract_indices(kwt1)
    main_2, second_2 = extract_indices(kwt2)
    if main_1 > main_2:
        return False
    if main_1 < main_2:
        return True
    elif second_1 < second_2:
        return True
    return False


class JsonParser:

    def __init__(self):
        self.web_topics = dict()
        self.clusters_lvl1 = dict()
        self.clusters_lvl2 = dict()
        self.web_topics_definition = dict()
        self.web_topics_duplicates = dict()
        self.file_mappings = dict()

    def process_json_files(self):
        logger.info('Loading web topics definitions')
        if len(self.web_topics_definition) == 0:
            with open(os.path.join(output_folder, 'web_topics_definitions.pkl'), 'rb') as input_file:
                self.web_topics_definition = pickle.load(input_file)

        logger.info('Begin processing json files...')

        idx = 0
        for root, dirs, files_list in os.walk(training_data_folder):
            logger.debug('root: {r}'.format(r=root))
            logger.debug('dirs: {d}'.format(d=dirs))
            # if idx > 50:
            #     break
            if not len(files_list) > 0:
                continue
            for json_file in files_list:
                if os.path.isfile(os.path.join(root, json_file)):
                    logger.debug("processing: %s/%s", root, json_file)
                    self.process_json(root, json_file)
                    idx += 1
                    if idx % 10 == 0:
                        logger.info("%s files processed...", idx)

        logger.info("Total nb files processed: %s", idx)
        logger.info('Nb topics level 1: %s' % len(self.web_topics))
        logger.info('Nb topics level 2: %s' % sum([len(lvl2) for lvl2 in self.web_topics]))
        logger.info('Nb files mapped: %s' % len(self.file_mappings))
        self.save_topics()
        self.save_mappings()
        self.save_clusters()

        logger.info('*'*30)
        logger.info('Topics')
        logger.info('-'*30)
        logger.info(pprint.pformat(self.web_topics))
        logger.info(pprint.pformat(self.web_topics))
        logger.info('-'*30)
        logger.info('Mappings')
        logger.info('-'*30)
        logger.info(pprint.pformat(self.file_mappings))
        logger.info('-'*30)
        logger.info('Clusters Level I')
        logger.info('-'*30)
        logger.info(pprint.pformat(self.clusters_lvl1))
        '''
        logger.info('-'*30)
        logger.info('Clusters Level II')
        logger.info('-'*30)
        logger.info(pprint.pformat(self.clusters_lvl2))
        '''
        logger.info('*' * 30)

    def process_json(self, root, json_file):
        with open(os.path.join(root, json_file), 'r', encoding='utf-8-sig') as json_in:
            json_obj = json.load(json_in)
            corpus_list = json_obj['corpusList']
            for item in corpus_list:
                lvl1 = item['levelOne']
                lvl2 = item['levelTwo']
                jt = item['jobTicketNumber']
                logger.debug("lvl1: {lvl1}".format(lvl1=lvl1))
                logger.debug("lvl2: {lvl2}".format(lvl2=lvl2))
                logger.debug("jt: {jt}".format(jt=jt))

                if lvl1 not in self.clusters_lvl1:
                    self.clusters_lvl1[lvl1] = list()
                if jt not in self.clusters_lvl1[lvl1]:
                    self.clusters_lvl1[lvl1].append(jt)
                if lvl2 not in self.clusters_lvl2:
                    self.clusters_lvl2[lvl2] = list()
                if jt not in self.clusters_lvl2[lvl2]:
                    self.clusters_lvl2[lvl2].append(jt)

                if lvl1 not in self.web_topics:
                    self.web_topics[lvl1] = list()
                if lvl2 not in self.web_topics_definition.keys():
                    continue
                if lvl2 not in self.web_topics[lvl1]:
                    self.web_topics[lvl1].append(lvl2)

                if jt not in self.file_mappings:
                    self.file_mappings[jt] = list()
                if lvl2 not in self.file_mappings[jt]:
                    self.file_mappings[jt].append(lvl2)

    def parse_topics_definitions(self):
        logger.info('Begin processing topics definitions...')
        with open(os.path.join(topics_definition_folder, 'KWT-List.json'), 'r', encoding='utf-8-sig') as json_in:
            json_obj = json.load(json_in)
            for kwt1 in json_obj:
                # pprint.pprint(json_obj[kwt1])
                if kwt1 not in self.web_topics_definition:
                    self.web_topics_definition[kwt1] = json_obj[kwt1]['label']
                if 'sublevels' in json_obj[kwt1]:
                    for kwt2 in json_obj[kwt1]['sublevels']:
                        if kwt2 not in self.web_topics_definition:
                            self.web_topics_definition[kwt2] = json_obj[kwt1]['sublevels'][kwt2]['label']
                        if 'duplicates' in json_obj[kwt1]['sublevels'][kwt2]:
                            if kwt2 not in self.web_topics_duplicates:
                                self.web_topics_duplicates[kwt2] = json_obj[kwt1]['sublevels'][kwt2]['duplicates']

            print('*' * 30)
            print('Topics Definition')
            print('-' * 30)
            pprint.pprint(self.web_topics_definition)
            print('-' * 30)
            print('Topics Duplicates')
            print('-' * 30)
            pprint.pprint(self.web_topics_duplicates)
            # self.remove_duplicates()
            print('-' * 30)
            with open(os.path.join(output_folder, 'web_topics_definitions.pkl'), 'wb') as out:
                pickle.dump(self.web_topics_definition, out)
            with open(os.path.join(output_folder, 'web_topics_duplicates.pkl'), 'wb') as out:
                pickle.dump(self.web_topics_duplicates, out)

    def remove_duplicates(self):
        for k, v in self.web_topics_duplicates.items():
            for dup in v:
                if is_lower(dup, k):
                    del self.web_topics_definition[k]
                    break

    def save_topics(self):
        with open(os.path.join(output_folder, 'web_topics.pkl'), 'wb') as out:
            pickle.dump(self.web_topics, out)

    def save_mappings(self):
        with open(os.path.join(output_folder, 'mappings.pkl'), 'wb') as out:
            pickle.dump(self.file_mappings, out)

    def save_clusters(self):
        with open(os.path.join(output_folder, 'clusters_lvl_I.pkl'), 'wb') as out:
            pickle.dump(self.clusters_lvl1, out)
        with open(os.path.join(output_folder, 'clusters_lvl_II.pkl'), 'wb') as out:
            pickle.dump(self.clusters_lvl2, out)


def main():
    json_parser = JsonParser()
    json_parser.parse_topics_definitions()
    json_parser.process_json_files()

if __name__ == '__main__':
    sys.exit(main())
