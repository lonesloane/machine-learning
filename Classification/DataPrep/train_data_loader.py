import json
import os
import sys
import pprint
from DataPrep import config, logger

training_data_folder = '/media/Data/OECD/Machine Learning/Mappings/Corpora'


class JsonParser:

    def __init__(self):
        self.topics = dict()
        self.file_mappings = dict()

    def process_json_files(self):

        logger.info('Begin processing json files...')

        idx = 0
        for root, dirs, files_list in os.walk(training_data_folder):
            logger.info('root: {r}'.format(r=root))
            logger.info('dirs: {d}'.format(d=dirs))
            # if idx > 50:
            #     break
            if not len(files_list) > 0:
                continue
            for json_file in files_list:
                if os.path.isfile(os.path.join(root, json_file)):
                    logger.info("processing: %s/%s", root, json_file)
                    self.process_json(root, json_file)
                    idx += 1
                    if idx % 10 == 0:
                        logger.info("%s files processed...", idx)

        logger.info("Total nb files processed: %s", idx)
        logger.info('Nb topics level 1: %s' % len(self.topics))
        logger.info('Nb topics level 2: %s' % sum([len(lvl2) for lvl2 in self.topics]))
        logger.info('Nb files mapped: %s' % len(self.file_mappings))
        #pprint.pprint(self.topics)
        #pprint.pprint(self.file_mappings)

    def process_json(self, root, json_file):
        with open(os.path.join(root, json_file), 'r', encoding='utf-8-sig') as json_in:
            json_obj = json.load(json_in)
            corpus_list = json_obj['corpusList']
            for item in corpus_list:
                lvl1 = item['levelOne']
                lvl2 = item['levelTwo']
                jt = item['jobTicketNumber']
                logger.info("lvl1: {lvl1}".format(lvl1=lvl1))
                logger.info("lvl2: {lvl2}".format(lvl2=lvl2))
                logger.info("jt: {jt}".format(jt=jt))
                if lvl1 not in self.topics:
                    self.topics[lvl1] = list()
                if lvl2 not in self.topics[lvl1]:
                    self.topics[lvl1].append(lvl2)
                if jt not in self.file_mappings:
                    self.file_mappings[jt] = list()
                if lvl2 not in self.file_mappings[jt]:
                    self.file_mappings[jt].append(lvl2)


def main():
    json_parser = JsonParser()
    json_parser.process_json_files()

if __name__ == '__main__':
    sys.exit(main())
