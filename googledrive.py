from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload,MediaFileUpload
from googleapiclient.discovery import build
import pprint
import io
import os.path

import gdown
import requests

import pandas as pd


class GoogleDrive:
    def __init__(self, client_secrets_path=None):
        self.id_file = ''
        self.url_file = f'https://drive.google.com/uc?id={self.id_file}'

        if os.path.isfile('c:/avid-grid-309317-687044cd90e9.json'):
            path_json = 'c:/avid-grid-309317-687044cd90e9.json'

        if os.path.isfile('avid-grid-309317-687044cd90e9.json'):
            path_json = 'avid-grid-309317-687044cd90e9.json'

        if client_secrets_path is not None:
            path_json = client_secrets_path

        credentials = service_account.Credentials.from_service_account_file(path_json)
        self.service = build('drive', 'v3', credentials=credentials)

        self._get_data()

        self.work_folder = self.df_main_folders['id'].iloc[0]

    def _get_id(self, url):
        return url.split('/')[5]

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
        self.df['parents1'] = self.df['parents'].apply(lambda x: x[0] if type(x) == list else x)
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

    def read_csv(self, name_file=None, id_file=None, url_share=None):
        if name_file is not None:
            id_file = self.df_nomain_folders[self.df_nomain_folders['name'] == name_file]['id'].iloc[0]

        if url_share is not None:
            id_file = self._get_id(url_share)

        url_file = f'https://drive.google.com/uc?id={id_file}'
        df = pd.read_csv(url_file)
        return df

    def set_work_folder(self, name_folder=None, id_folder=None, url_share=None):
        if name_folder is not None:
            self.work_folder = self.df[(self.df['name'] == name_folder) & (self.df['mimeType'] == 'application/vnd.google-apps.folder')]['id'].iloc[0]

        if url_share is not None:
            id_folder = self._get_id(url_share)

        if id_folder is not None:
            self.work_folder = id_folder

    def download_file(self, id_file, name_file=None):

        request = self.service.files().get_media(fileId=id_file)

        if name_file is None:
            filename = self.df[self.df['id'] == id_file]['name'].iloc[0]

        with io.FileIO(filename, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Download %d%%." % int(status.progress() * 100))

    def upload_file(self, file_path, name_file_new=None, folder_id=None):

        if folder_id is None:
            folder_id = self.work_folder

        if name_file_new is None:
            name_file_new = file_path

        file_metadata = {
            'name': name_file_new,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        r = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return r