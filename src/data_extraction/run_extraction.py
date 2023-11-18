from os import path, listdir
from reader import file_types, read_image_and_write

folder_path = '../../foto/'
used_photo_file = folder_path + 'used_photo.txt'

def run_extraction():

    new_photos_found = False

    for filename in listdir(folder_path):
        if filename.endswith(file_types):
            file_path = path.join(folder_path, filename)
            img, new_photo_flag = read_image_and_write(file_path, used_photo_file)
            if new_photo_flag: new_photos_found = True
            
    if new_photos_found:
        print(f"Processing completed. Names of successfully opened images written to {used_photo_file}")
    else:
        print("No new photos found.")
        
if __name__ == "__main__": run_extraction()
