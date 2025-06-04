import logging
import os

class Logger:
    def __init__(self, log_dir, log_file='log.txt'):
        """
        初始化日志记录器
        :param log_dir: 日志存储目录
        :param log_file: 日志文件名
        """
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        self.logger = logging.getLogger('MambaUNetRainRemoval')
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)

    def info(self, message):
        """记录信息级别日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告级别日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误级别日志"""
        self.logger.error(message)