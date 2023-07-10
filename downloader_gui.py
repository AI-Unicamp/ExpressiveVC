from inference_gui2 import MODELS_DIR
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QCheckBox, QTableWidget,
    QApplication, QFrame, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QAbstractScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
import huggingface_hub
import os
import glob
import shutil
import sys
import requests
from pathlib import Path

# Only enable this if you plan on training off a downloaded model.
DOWNLOAD_DISCRIMINATORS = False
MODELS_DIR = os.path.join("so-vits-svc",MODELS_DIR)
class DownloadStrategy:
    def __init__(self, repo_id : str, model_dir : str):
        """ Pull from HF to find available models """
        pass

    def get_available_model_names(self) -> list:
        """ Returns a list of model names """
        pass

    def check_present_model_name(self, name : str) -> bool:
        """ Returns True if model is already installed """
        return False

    def download_model(self, name : str):
        """ Downloads model corresponding to name """
        pass

class FolderStrategy(DownloadStrategy):
    def __init__(self, repo_id, model_dir):
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.model_repo = huggingface_hub.Repository(
            local_dir=self.model_dir, clone_from=self.repo_id,
            skip_lfs_files=True)
        self.model_repo.git_pull(lfs=False)
        self.model_folders = os.listdir(model_dir)
        self.model_folders.remove('.git')
        self.model_folders.remove('.gitattributes')

    def get_available_model_names(self):
        return self.model_folders

    def check_present_model_name(self, name):
        return bool(name in os.listdir(MODELS_DIR))

    def download_model(self, model_name):
        print("Downloading "+model_name)
        basepath = os.path.join(self.model_dir, model_name)
        targetpath = os.path.join(MODELS_DIR, model_name)
        gen_pt = next(x for x in os.listdir(basepath) if x.startswith("G_"))
        disc_pt = next(x for x in os.listdir(basepath) if x.startswith("D_"))
        cfg = next(x for x in os.listdir(basepath) if x.endswith("json"))
        clust = [x for x in os.listdir(basepath) if x.endswith("pt")]

        huggingface_hub.hf_hub_download(repo_id = self.repo_id,
            filename = model_name + "/" + gen_pt, local_dir = MODELS_DIR,
                cache_dir = './cache', local_dir_use_symlinks=False,
                force_download=True)

        if DOWNLOAD_DISCRIMINATORS:
            huggingface_hub.hf_hub_download(repo_id = self.repo_id,
                filename = model_name + "/" + disc_pt, local_dir = MODELS_DIR,
                cache_dir = './cache', local_dir_use_symlinks=False,
                force_download=True)
        if len(clust) != 0:
            for c in clust:
                huggingface_hub.hf_hub_download(repo_id = self.repo_id,
                    filename = model_name + "/" + c, local_dir = MODELS_DIR,
                        cache_dir = './cache', local_dir_use_symlinks=False,
                        force_download=True)
        shutil.copy(os.path.join(basepath, cfg), os.path.join(targetpath, cfg))

from zipfile import ZipFile

class ZipStrategy(DownloadStrategy):
    def __init__(self, repo_id, model_dir, is_model=True):
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.is_model = is_model # model or dataset
        self.model_repo = huggingface_hub.Repository(
            local_dir=self.model_dir, clone_from=self.repo_id,
            skip_lfs_files=True, repo_type=("model" if is_model else "dataset")
            )
        self.model_repo.git_pull(lfs=False)
        self.model_zips = glob.glob(model_dir + "/**/*.zip", recursive=True)

        self.model_names = [Path(x).stem for x in self.model_zips]
        self.rel_paths = {Path(x).stem :
            str(Path(x).relative_to(model_dir)).replace('\\','/')
                for x in self.model_zips}

    def get_available_model_names(self):
        return self.model_names

    def check_present_model_name(self, name):
        return bool(name in os.listdir(MODELS_DIR))

    def download_model(self, model_name):
        print("Downloading "+self.rel_paths[model_name])
        huggingface_hub.hf_hub_download(repo_id = self.repo_id,
            filename = self.rel_paths[model_name], local_dir = MODELS_DIR,
                cache_dir="./cache", local_dir_use_symlinks=False,
                force_download=True, repo_type=(
                    "model" if self.is_model else "dataset"))

        zip_path = os.path.join(MODELS_DIR,self.rel_paths[model_name])
        with ZipFile(zip_path, 'r') as zipObj:
            zip_contents = zipObj.namelist()
            print(os.path.dirname(zip_contents[0]))
            if len(zip_contents) and len(
                os.path.dirname(zip_contents[0])) > 0:
                # assume that this zip is structured with 1 or more folders
                # containing speaker models directly
                zipObj.extractall(MODELS_DIR)
            else:
                # assume that this zip contains speaker models directly
                zipObj.extractall(os.path.join(MODELS_DIR, model_name))
        os.remove(zip_path)

        # clean stub directories
        for root, dirs, files in os.walk(MODELS_DIR, topdown=False):
            if not dirs and not files:
                os.rmdir(root)


class UrlZipStrategy(DownloadStrategy):
    def __init__(self, repo_id=None, model_dir=None):
        self.model_urls = ["https://huggingface.co/datasets/HazySkies/SV3/"
            "resolve/main/sovits_athena_44khz_10000_sv4.zip",
            "https://huggingface.co/datasets/HazySkies/SV3/resolve/main/"
            "sovits_athena_44khz_25000_sv4.zip",
            "https://huggingface.co/datasets/HazySkies/SV3/resolve/main/"
            "sovits_tfh_arizona_44khz_20000_sv4.zip",
            "https://huggingface.co/datasets/HazySkies/SV3/resolve/main"
            "sovits_tfh_velvet_44khz_20000_sv4.zip"]

        self.model_names = [Path(x).stem for x in self.model_urls]
        self.name_url_map = { Path(x).stem : x for x in self.model_urls }

    def get_available_model_names(self):
        return self.model_names

    def check_present_model_name(self, name):
        return bool(name in os.listdir(MODELS_DIR))

    def download_model(self, model_name):
        print("Downloading "+model_name)
        zip_path = os.path.join(MODELS_DIR,model_name+'.zip')
        zip_url = self.name_url_map[model_name]
        r = requests.get(zip_url, allow_redirects=True)
        with open(zip_path) as f:
            f.write(r.content)
        with ZipFile(zip_path, 'r') as zipObj:
            zipObj.extractall(MODELS_DIR)
        os.remove(zip_path)

class DownloaderGui (QMainWindow):
    def __init__(self):
        super().__init__()
        print("Downloading repos...")
        self.strategies = [
            FolderStrategy("therealvul/so-vits-svc-4.0",
                "repositories/hf_vul_model"),
            FolderStrategy("OlivineEllva/so-vits-svc-4.0-models",
                "repositories/hf_oe_model"),
            ZipStrategy("Amo/so-vits-svc-4.0_GA",
                "repositories/hf_amo_models"),
            ZipStrategy("HazySkies/SV3",
                "repositories/hf_hazy_models", False)]
        print("Finished downloading repos")

        self.setWindowTitle("so-vits-svc 4.0 Downloader")
        self.central_widget = QFrame()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.model_table = QTableWidget()
        self.layout.addWidget(self.model_table)
        self.model_table.setColumnCount(3)
        self.model_table.setHorizontalHeaderLabels(
            ['Model name', 'Check to install', 'Detected?'])
        self.model_table.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents)

        self.available_models = {}
        self.present_map = {}
        self.checkbox_map = {}
        for i,v in enumerate(self.strategies):
            available_models = v.get_available_model_names()
            for m in available_models:
                self.available_models[m] = i
                self.present_map[m] = v.check_present_model_name(m)
                self.checkbox_map[m] = QCheckBox()
                self.checkbox_map[m].setStyleSheet("margin-left:50%;"
                                                   "margin-right:50;")

        # Populate table
        self.model_table.setRowCount(len(self.available_models.items()))
        for i,v0 in enumerate(self.available_models.items()):
            k,v = v0
            # Model name
            model_name = QTableWidgetItem(str(k))
            self.model_table.setItem(i,0,model_name)
            model_name.setFlags(model_name.flags() ^ Qt.ItemIsEditable)
            # Check to install
            self.model_table.setCellWidget(i,1,self.checkbox_map[k])
            # Detected on system?
            detected = QTableWidgetItem(str(self.present_map[k]))
            detected.setFlags(detected.flags() ^ Qt.ItemIsEditable)
            self.model_table.setItem(i,2,detected)

        self.model_table.resizeColumnsToContents()
        self.model_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.model_table.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)

        size = QSize(600,1000)
        self.model_table.setMinimumSize(size)
        self.model_table.setMaximumSize(size)

        self.download_button = QPushButton("Download selected")
        self.layout.addWidget(self.download_button)
        self.download_button.clicked.connect(self.download_selected)

    def download_selected(self):
        for i,v0 in enumerate(self.available_models.items()):
            k,v = v0
            if self.checkbox_map[k].isChecked():
                self.strategies[self.available_models[k]].download_model(k)
        self.update_available()

    def update_available(self):
        self.model_table.clearContents()

        for i,v0 in enumerate(self.available_models.items()):
            k,v = v0
            self.present_map[k] = self.strategies[v].check_present_model_name(
                k)

            # Model name
            model_name = QTableWidgetItem(str(k))
            self.model_table.setItem(i,0,model_name)
            model_name.setFlags(model_name.flags() ^ Qt.ItemIsEditable)
            # Check to install
            self.checkbox_map[k] = QCheckBox()
            self.checkbox_map[k].setStyleSheet("margin-left:50%;"
                                               "margin-right:50;")
            self.model_table.setCellWidget(i,1,self.checkbox_map[k])
            # Detected on system?
            detected = QTableWidgetItem(str(self.present_map[k]))
            detected.setFlags(detected.flags() ^ Qt.ItemIsEditable)
            self.model_table.setItem(i,2,detected)

if __name__ == '__main__':
    if (Path(os.getcwd()).stem == 'so-vits-svc' or 
        ("inference_gui2.py" in os.listdir())):
        os.chdir('..')

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    app = QApplication(sys.argv)
    w = DownloaderGui()
    w.show()
    app.exec()
