from os import (
    path, listdir
)

from data_extraction.reader import (
    file_types, read_image_and_write
)

from data_extraction.feature_extraction import feature_extraction

from numpy import ndarray

from config import paths

foto_folder_path = paths.foto_folder_path
used_photo_file = paths.used_photo_file

def run_extraction():

    new_photos_found = False

    for filename in listdir(foto_folder_path):
        if filename.endswith(file_types):
            file_path = path.join(foto_folder_path, filename)
            img, new_photo_flag = read_image_and_write(file_path, used_photo_file)
            if type(img) == ndarray: feature_extraction(image=img, image_name=filename)
            if new_photo_flag: new_photos_found = True
            
    if new_photos_found:
        print(f"Processing completed. Names of successfully opened images written to {used_photo_file}")
    else:
        print("No new photos found.")
