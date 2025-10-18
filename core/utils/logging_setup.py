"""
Logging for verbose CLI diagnosis
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

#Environment variables
DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() #defaults to logging.INFO
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s" #timestamp - level - name - message
DEFAULT_DATEFMT = "%H:%M:%S" #24 hour clock
DEFAULT_LOGFILE = os.getenv("LOG_FILE", "../../app.log") #logging file

def configure(level: str | int = DEFAULT_LEVEL, to_file: bool = True, filename: str = DEFAULT_LOGFILE):
    """
    Configure root logger once. Safe to call multiple times; it will no-op if already configured.
    - level: string like "DEBUG"/"INFO"/... or numeric level
    - to_file: log to a file or not
    - filename: log file name (default: app.log)
    """
    #Convert string level to numeric if needed
    #Ex: logging.DEBUG == 10, INFO = 20, WARNING = 30, etc.
    #Falls back on logging.INFO if an unknown string is passed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    #Avoid duplicate handlers; get root logger
    root = logging.getLogger()
    if root.handlers:
        return
    
    #Minimum severity to process
    root.setLevel(level)
    
    #Log format
    formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT)

    #Console handler; handler created to write output, then apply formatter, then attach to the root logger
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    #Optional file handler (rotates ~1MB, keeps 3 backups)
    #File rotation; replacing saved log file based on conditions
    #If exceeds 1 MB, move onto another file (up to 3 maximum)
    if to_file:
        file_handler = RotatingFileHandler(filename, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """Wrapper to get logger's name"""
    return logging.getLogger(name)