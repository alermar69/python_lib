from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload,MediaFileUpload
from googleapiclient.discovery import build
import pprint
import io

import gdown
import requests

import pandas as pd


class Googledrive:
    def __init__(self):
        self.id_file = ''
        self.url_file = f'https://drive.google.com/uc?id={self.id_file}'
        self.url_json = 'https://raw.githubusercontent.com/alermar69/python_lib/main/total-pier-309101-0350868dbf27.json?token=ADJ5RTD4H7S4XRDEUQRQUKDAMHLOM'

        r = requests.get(self.url_json)
        credentials = service_account.Credentials.from_service_account_info(r.json())
        self.service = build('drive', 'v3', credentials=credentials)

        self._get_data()

    def _get_data(self):
        results = self.service.files().list(
            pageSize=10,
            fields="nextPageToken, files(id, name, mimeType, parents, createdTime)").execute()

        nextPageToken = results.get('nextPageToken')
        while nextPageToken:
            nextPage = self.service.files().list(pageSize=10,
                                            fields="nextPageToken, files(id, name, mimeType, parents, createdTime)",
                                            pageToken=nextPageToken).execute()
            nextPageToken = nextPage.get('nextPageToken')
            results['files'] = results['files'] + nextPage['files']

        self.df = pd.DataFrame(results.get('files'))
        self.df_main_folders = self.df.loc[(pd.isnull(self.df['parents'])) & (self.df['mimeType'] == 'application/vnd.google-apps.folder')]
        self.df_nomain_folders = self.df.loc[~pd.isnull(self.df['parents'])]

    def print_files(self):
        for folds in self.df_main_folders[['id', 'name']].values:
            self._print_files(folds[0], folds[1], 0)
            print('-' * 100)

    def _print_files(self, id_fold, name, i):
        print('\t' * i, name)
        df3 = self.df_nomain_folders[
            (self.df_nomain_folders['parents'].apply(lambda x: id_fold in x)) & (self.df_nomain_folders['mimeType'] != 'application/vnd.google-apps.folder')]
        print(df3['name'].values)

        df4 = self.df_nomain_folders[
            (self.df_nomain_folders['parents'].apply(lambda x: id_fold in x)) & (self.df_nomain_folders['mimeType'] == 'application/vnd.google-apps.folder')]
        if len(df4) != 0:
            for folds in df4[['id', 'name']].values:
                self._print_files(folds[0], folds[1], i + 1)
            return 1
        else:
            return 0

    def get_file(self, name, id_file=None):
        pass

    def read_csv(self, name, id_file=None):
        if id_file is None:
            id_file = self.df_nomain_folders[self.df_nomain_folders['name'] == name]['id'].iloc[0]

        url_file = f'https://drive.google.com/uc?id={id_file}'
        df = pd.read_csv(url_file)
        return df

    def load_file(self, name):
        results = self.service.files().list(
            pageSize=10,
            fields="nextPageToken, files(id, name, mimeType, parents, createdTime)",
            q="'1uuecd6ndiZlj3d9dSVeZeKyEmEkC7qyr' in parents and name contains 'data'").execute()

        file_id = results.get('files')[0].get('id')
        request = self.service.files().get_media(fileId=file_id)
        filename = 'german_credit_data1.csv'

        with io.FileIO(filename, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Download %d%%." % int(status.progress() * 100))