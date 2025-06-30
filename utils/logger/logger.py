import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import logging.config
import logging.handlers
import json
import os


mylogger = logging.getLogger("ink_agent")

config_file = pathlib.Path("utils/logger/config.json")
with open(config_file) as file:
    config = json.load(file)
# create log directory for file handler

os.makedirs("logs", exist_ok=True)
logging.config.dictConfig(config)
logging.basicConfig(level="INFO")

if __name__ == "__main__":
    mylogger.info("info test")
    mylogger.debug("debug test")
    mylogger.critical("crital test")
    mylogger.error("error test")
   

