"""
File containing logging set up.
"""
import os
import logging


def logging_setup(log_dir: str) -> None:
    """
    Function which sets up logging configuration.
    Logs will be saved in log.txt and displayed on stderr.

    Parameters
    ----------
    log_dir: str
        Path where to store log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')
    filemode = 'a' if os.path.exists(log_path) else 'w'

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
