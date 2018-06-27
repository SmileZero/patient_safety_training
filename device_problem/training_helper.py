from .device_problem_scrapper import DeviceProblemScrapper
import threading
from django.conf import settings


class TrainingHelper(threading.Thread):
    def __init__(self, start_report_key, end_report_key):
        self.start_report_key = start_report_key
        self.end_report_key = end_report_key
        threading.Thread.__init__(self)

    def run(self):
        status_logger = open(settings.TRAINING_LOG_FILE, 'w', 1)  # 1 means line buffer
        status_logger.write("Start Training...\n")

        status_logger.write("Start Scrapping Device Problem..\n")

        if self.end_report_key > self.start_report_key:
            scrapper = DeviceProblemScrapper(self.start_report_key + 1, self.end_report_key, status_logger)
            scrapper.get_data()

        status_logger.write("Scraping Done\n")

        # TODO: get noise
        status_logger.write("Start Getting Noise From Solr...\n")

        status_logger.write("Getting Noise Done\n")

        # TODO: run training
        status_logger.write("Start Training...\n")

        status_logger.write("Training Done\n")
        status_logger.close()