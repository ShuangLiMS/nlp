import logging
import datetime
import os

def logger(module_prefix, logging_folder=None):
    if logging_folder is not None:
        logger = logging.getLogger()

        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        time_string = datetime.datetime.utcnow().strftime("_%m-%d-%y:%H-%M-%S")
        fh = logging.FileHandler(os.path.join(logging_folder, module_prefix + time_string + ".log"))
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

    logger = logging.getLogger(module_prefix)
    return logger
