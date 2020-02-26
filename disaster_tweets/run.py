import logging
from logging.handlers import RotatingFileHandler

from app import app


if __name__ == '__main__':
    handler = RotatingFileHandler('disaster.log', maxBytes=100000, backupCount=0)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run()
