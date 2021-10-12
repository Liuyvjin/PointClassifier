import logging


class LogTool():
    """ 日志工具类 """
    def __init__(self, file_path, format=None):
        self.logger = logging.getLogger("Default")
        self.logger.setLevel(logging.INFO)

        if format:
            formatter = logging.Formatter(format)
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def cprint(self, text):
        print(text)
        self.logger.info(text)

    def info(self, text):
        self.logger.info(text)