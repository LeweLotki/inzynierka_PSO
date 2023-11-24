from os import path
from cv2 import imread

file_types = ('.png', '.jpg', '.jpeg', '.bmp')

def read_image_and_write(file_path: str, output_file: str) -> tuple:
    
    _, filename = path.split(file_path)
    
    if not path.isfile(output_file):
        # If used_photo.txt doesn't exist, create an empty file
        open(output_file, 'a').close()
    
    with open(output_file, 'r') as file:
        used_photos = file.read().splitlines()

    if filename in used_photos:
        # File name already exists in used_photo.txt
        return None, False
    else: print(f"'{filename}' \n")

    image = imread(file_path)
    if image is not None:
        with open(output_file, 'a') as file:
            file.write(filename + '\n')
        return image, True
    else:
        print(f"Error: Unable to open image at '{file_path}'")
        return None, False
