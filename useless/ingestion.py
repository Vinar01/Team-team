import io
import os
import pandas as pd
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential

class SharePointDownloader:
    def __init__(self, site_url, username, password):
        """Authenticates with the SharePoint site."""
        self.ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))

    def fetch_csv_to_dataframe(self, relative_file_url):
        """Downloads a CSV directly into a pandas DataFrame."""
        try:
            response = self.ctx.web.get_file_by_server_relative_url(relative_file_url).download()
            csv_content = io.BytesIO(response.content)
            df = pd.read_csv(csv_content)
            print(f"Successfully loaded CSV: {relative_file_url}")
            return df
        except Exception as e:
            print(f"Error fetching CSV {relative_file_url}: {e}")
            return None

    def download_audio_file(self, relative_file_url, download_dir="temp_audio"):
        """Downloads an audio file to a local directory."""
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        file_name = os.path.basename(relative_file_url)
        local_path = os.path.join(download_dir, file_name)
        
        try:
            with open(local_path, "wb") as local_file:
                self.ctx.web.get_file_by_server_relative_url(relative_file_url).download(local_file)
            return local_path
        except Exception as e:
            print(f"Error downloading audio {relative_file_url}: {e}")
            return None
