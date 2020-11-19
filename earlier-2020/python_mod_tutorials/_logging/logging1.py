# -*- coding: utf-8 -*-
"""

# Log record to a file
logging.basicConfig(filename='logging.txt')

# more complex
logging.getLogger()  # etc

"""

import logging


logging.basicConfig(level=logging.INFO)

# Logging rank
logging.debug('Hello world!')
logging.info('Hello world!')
logging.warning('Hello world!')
logging.error('Hello world!')
logging.critical('Hello world!')