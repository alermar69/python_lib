from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload,MediaFileUpload
from googleapiclient.discovery import build
import pprint
import io

import gdown
import requests


class Googledrive:
    def __init__(self):
        self.id_file = ''
        self.url_file = f'https://drive.google.com/uc?id={self.id_file}'
        self.url_json = 'https://raw.githubusercontent.com/alermar69/python_lib/main/total-pier-309101-0350868dbf27.json?token=ADJ5RTD4H7S4XRDEUQRQUKDAMHLOM'

        r = requests.get(self.url_json)
        credentials = service_account.Credentials.from_service_account_info(r.json())
        self.service = build('drive', 'v3', credentials=credentials)

    def get_file(self, name):
        results = self.service.files().list(pageSize=10,
                                       fields="nextPageToken, files(id, name, mimeType)").execute()

        nextPageToken = results.get('nextPageToken')
        while nextPageToken:
            nextPage = self.service.files().list(pageSize=10,
                                            fields="nextPageToken, files(id, name, mimeType, parents)",
                                            pageToken=nextPageToken).execute()
            nextPageToken = nextPage.get('nextPageToken')
            results['files'] = results['files'] + nextPage['files']

        return results

    def down_file(self, name):
        pass