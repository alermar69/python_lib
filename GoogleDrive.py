from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload,MediaFileUpload
from googleapiclient.discovery import build
import pprint
import io

import gdown


class Googledrive:
    def __init__(self):
        self.id_file = ''
        self.url_file = f'https://drive.google.com/uc?id={self.id_file}'

    def get_file(self, name):
        pass