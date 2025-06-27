import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import logging.config
import logging.handlers
import json


mylogger = logging.getLogger("ink_agent")

config_file = pathlib.Path("logger/config.json")
with open(config_file) as file:
    config = json.load(file)
logging.config.dictConfig(config)
logging.basicConfig(level="INFO")

if __name__ == "__main__":
    mylogger.info("info test")
    mylogger.debug("debug test")
