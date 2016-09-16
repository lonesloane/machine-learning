# TODO: move to module level
# Get configuration parameters
import logging
import os
from configparser import ConfigParser

basedir = os.path.abspath(os.path.dirname(__file__))
config = ConfigParser()
config.read(os.path.join(basedir, 'framer.conf'))

# Set appropriate logging level
numeric_level = getattr(logging, config.get('LOGGING', 'level').upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % config.get('LOGGING', 'level'))
logger = logging.getLogger(__name__)
logger.setLevel(numeric_level)
# create file handler which logs even debug messages
fh = logging.FileHandler(config.get('LOGGING', 'log_file'), mode='w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
