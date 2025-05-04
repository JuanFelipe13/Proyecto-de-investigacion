import os
import urllib.request
import tarfile
from tqdm import tqdm
import shutil

DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DOWNLOAD_PATH = "food-101.tar.gz"
EXTRACT_PATH = "data/"

def download_with_progress(url, filename):
    print(f"Descargando {filename}...")
    response = urllib.request.urlopen(url)
    total_size = int(response.headers['Content-Length'])
    
    with open(filename, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        while True:
            data = response.read(1024)
            if not data:
                break
            f.write(data)
            pbar.update(len(data))

def organize_dataset():
    print("Organizando dataset...")
    
    os.makedirs("data/food-101/train", exist_ok=True)
    os.makedirs("data/food-101/test", exist_ok=True)
    
    with open("data/food-101/meta/train.txt", "r") as f:
        train_files = [line.strip() for line in f.readlines()]
    
    with open("data/food-101/meta/test.txt", "r") as f:
        test_files = [line.strip() for line in f.readlines()]
    
    for file_path in tqdm(train_files, desc="Organizando train"):
        class_name = file_path.split("/")[0]
        os.makedirs(f"data/food-101/train/{class_name}", exist_ok=True)
        src = f"data/food-101/images/{file_path}.jpg"
        dst = f"data/food-101/train/{class_name}/{file_path.split('/')[1]}.jpg"
        shutil.copy(src, dst)
    
    for file_path in tqdm(test_files, desc="Organizando test"):
        class_name = file_path.split("/")[0]
        os.makedirs(f"data/food-101/test/{class_name}", exist_ok=True)
        src = f"data/food-101/images/{file_path}.jpg"
        dst = f"data/food-101/test/{class_name}/{file_path.split('/')[1]}.jpg"
        shutil.copy(src, dst)

def main():
    if not os.path.exists(DOWNLOAD_PATH):
        download_with_progress(DATASET_URL, DOWNLOAD_PATH)
    
    if not os.path.exists("data/food-101"):
        print("Extrayendo archivo...")
        with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
            tar.extractall(path=EXTRACT_PATH)
    
    organize_dataset()
    
    if os.path.exists(DOWNLOAD_PATH):
        os.remove(DOWNLOAD_PATH)
    
    print("Dataset preparado correctamente!")

if __name__ == "__main__":
    main() 