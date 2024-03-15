from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Specify file id on google drive for download
drive_ids = {
    "dlibshape_predictor_68_face_landmarks.dat": "1-AC9iWOMEJVloP3O6Mt1JVDbyK7ABK87",
    "e4e_ffhq_encode.pt": "1-0PBbFDSMyNtPOPA7qv_hFDQcs-6Z3le",
    "jojo.pt": "1-WMukQ2FaWzrnUF09ryg0iDLllpEMBP3",
    "stylegan2-ffhq-config-f.pt": "1-2pVEfLMNqeiZM1elLqCy6FcpJp5mITd",
}


class Downloader(object):
    def __init__(self, use_pydrive):
        self.use_pydrive = use_pydrive

        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, file_name):
        file_dst = os.path.join(os.getcwd(), file_name)
        file_id = drive_ids[file_name]
        print(f'Downloading {file_name}')
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id':file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            !gdown --id $file_id -O $file_dst

download_with_pydrive = True
downloader = Downloader(download_with_pydrive)

downloader.download_file('dlibshape_predictor_68_face_landmarks.dat')
downloader.download_file('e4e_ffhq_encode.pt')
downloader.download_file('jojo.pt')
downloader.download_file('stylegan2-ffhq-config-f.pt')
