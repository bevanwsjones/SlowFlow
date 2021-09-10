# ----------------------------------------------------------------------------------------------------------------------
#  This file is part of the SlowFlow distribution  (https://github.com/bevanwsjones/SlowFlow).
#  Copyright (c) 2020 Bevan Walter Stewart Jones.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, version 3.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see
#  <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------------------------------------------------
# filename: logging
# description: Contains methods to handle the console and file logging.
# ----------------------------------------------------------------------------------------------------------------------

import logging.config
import logging
import os

LOGGING_FOLDER = './session'
LOGGING_FILE = 'session.log'

LOG_LEVEL_DICTIONARY = dict(
    debug=logging.DEBUG,
    info=logging.INFO,
    warning=logging.WARNING,
    error=logging.ERROR,
    critical=logging.CRITICAL,
)

COLOR_DICTIONARY = dict(
    none='',
    black='30',
    red='31',
    green='32',
    yellow='33',
    blue='34',
    magenta='35',
    cyan='36',
    white='37',
    bright_black='90',
    bright_red='91',
    bright_green='92',
    bright_yellow='93',
    bright_blue='94',
    bright_magenta='95',
    bright_cyan='96',
    bright_white='97'
)

EMPHASIS_DICTIONARY = dict(
    none='',
    reset='0',
    bold='1',
    faint='2',
    italic='3',
    underline='4',
    strike='9',
    normal='22'
)

LOG_RECORD_ATTRIBUTES_DICTIONARY = dict(
    asctime='%(asctime)s',
    created='%(created)f',
    file_name='%(filename)s',
    function_name='%(funcName)s',
    level_name='%(levelname)-8s',
    level_number='%(levelno)s',
    line_number='%(lineno)d',
    message='%(message)s',
    module='%(module)s',
    msecs='%(msecs)d',
    name='%(name)s',
    path_name='%(pathname)s',
    process='%(process)d',
    process_Name='%(processName)s',
    relative_Created='%(relativeCreated)d',
    thread='%(thread)d',
    thread_Name='%(threadName)s'
)


def build_text_format(_text, _color='none', _emphasis='none'):
    """
    Builds a formatted string, text, for output to the terminal or file using ANSI escape characters.
    Note: ANSI cannot be used with utf8 files, so writing of files should not include color or emphasis strings.
    :param _text: String containing the text being formatted
    :type _text: str
    :param _color: The color dictionary look up string, defaults to none.
    :type _color: str
    :param _emphasis: The emphasis dictionary look up string, defaults to none.
    :type _emphasis: str
    :return: The inputted string correctly formatted.
    :type: str
    """
    _is_format = _color != 'none' or _emphasis != 'none'  # short hand for if we are formatting
    string = '\x1b[' if _is_format else ''
    string += COLOR_DICTIONARY[_color]
    string += ';' if _is_format and _emphasis != 'none' else ''
    string += EMPHASIS_DICTIONARY[_emphasis]
    string += 'm' if _is_format else ''
    string += _text
    string += '\x1b[0m' if _is_format else ''

    return string


def build_log_format(_attribute, _color='none', _emphasis='none'):
    """
    Wrapper of build_text_format, format default logger 'named' attributes.
    Note: ANSI cannot be used with utf8 files, so writing of files should not include color or emphasis strings.
    :param _attribute: The logger attribute dictionary look up string.
    :type _attribute: str
    :param _color: The color dictionary look up string, defaults to none.
    :type _color: str
    :param _emphasis: The emphasis dictionary look up string, defaults to none.
    :type _emphasis: str
    :return: The inputted string correctly formatted.
    :type: str
    """
    return build_text_format(LOG_RECORD_ATTRIBUTES_DICTIONARY[_attribute], _color, _emphasis)


class StreamFormatter(logging.Formatter):
    """
    Formatter for console logging, creates a full logging string which is colorized and emphasised.
    """
    space_character = build_text_format(' : ', 'bright_white')
    time_string = build_log_format('asctime', 'bright_white')
    name_string = build_log_format('name', _color='magenta')
    message_string = build_log_format('message', 'white', 'normal')

    FORMATS = (
        {
            logging.DEBUG: time_string + space_character + build_log_format('level_name', 'bright_blue', 'bold')
                           + space_character + name_string + space_character + message_string,
            logging.INFO: time_string + space_character + build_log_format('level_name', 'green', 'bold') + space_character
                          + name_string + space_character + message_string,
            logging.WARNING: time_string + space_character + build_log_format('level_name', 'yellow', 'bold') + space_character + name_string + space_character + message_string,
            logging.ERROR: time_string + space_character + build_log_format('level_name', 'red', 'bold') + space_character
                           + name_string + space_character + message_string,
            logging.CRITICAL: time_string + space_character + build_log_format('level_name', 'bright_red', 'bold') + space_character + name_string + space_character + message_string
        }
    )

    def format(self, _record):
        """
        Returns a formatted string depending on the record level.
        Note: this function is for use inside the python logger.
        :param _record: Instance of the message being logged.
        :type _record: logging.LogRecord
        """
        return logging.Formatter(self.FORMATS.get(_record.levelno)).format(_record)


class FileFormatter(logging.Formatter):
    """
    Formatter for file logging, creates a full logging string which is not colorized or emphasised due to file being
     writen with utf8.
    """

    space_character = build_text_format(':')
    time_string = build_log_format('asctime')
    name_string = build_log_format('name')
    message_string = build_log_format('message')

    FORMATS = {
        logging.DEBUG: time_string + space_character + build_log_format('level_name') + space_character + name_string +
                       space_character + message_string,
        logging.INFO: time_string + space_character + build_log_format('level_name') + space_character + name_string +
                      space_character + message_string,
        logging.WARNING: time_string + space_character + build_log_format('level_name') + space_character + name_string
                         + space_character + message_string,
        logging.ERROR: time_string + space_character + build_log_format('level_name') + space_character + name_string +
                       space_character + message_string,
        logging.CRITICAL: time_string + space_character + build_log_format('level_name') + space_character + name_string
                          + space_character + message_string
    }

    def format(self, _record):
        """
        Returns a formatted string depending on the record level.
        Note: this function is for use inside the python logger.
        :param _record: Instance of the message being logged.
        :type _record: logging.LogRecord
        """
        return logging.Formatter(self.FORMATS.get(_record.levelno)).format(_record)


def set_up_logging():
    """
    Sets up the logging directory and file, buy creating a session directory if none exists and deleting the old
    session.log file if it exists.
    """

    if os.path.isdir(LOGGING_FOLDER):
        if os.path.exists(LOGGING_FOLDER + '/' + LOGGING_FILE):
            # more robust than actualy trying to delete the file.
            with open(LOGGING_FOLDER + '/' + LOGGING_FILE, "w") as log_file:
                log_file.write('')
            log_file.close()
    else:
        raise FileNotFoundError('Could not file folder: ' + LOGGING_FOLDER + ' when trying to clear ' + LOGGING_FILE)


def make_logger(_name, _level):
    """
    Creates a new logger with both console and file handlers set up. The console handler will have colorized text, while
    the file handler will write unformated text to the
     './session/session.log' file.

    :param _name: Name of the new logger, it is highly recommended to use __name__.
    :type _name: str
    :param _level: The log level dictionary look up string, used to omit certain messages depending on the severity of
                   the message and level chosen.
    :type _level: str
    :return: Returns a new logger
    :type: logging.Logger
    """

    # setup logger
    logger = logging.getLogger(_name)
    logger.setLevel(LOG_LEVEL_DICTIONARY[_level])

    # setup stream handler
    steam_handler = logging.StreamHandler()
    steam_handler.setFormatter(StreamFormatter())
    logger.addHandler(steam_handler)

    # setup file handler
    try:
        file_handler = logging.FileHandler(LOGGING_FOLDER + '/' + LOGGING_FILE, encoding='utf8')
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)
    except FileNotFoundError as e:
        logger.warning(e)
        logger.warning("Logger now only logging to console")
        pass

    return logger
