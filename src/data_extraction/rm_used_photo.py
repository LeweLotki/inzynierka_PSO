from os import (
    remove, path
)

def rm_used_photo():
    used_photo_file = '../foto/used_photo.txt'

    if path.exists(used_photo_file):
        remove(used_photo_file)
        
    print('remove "used_photo.txt"')
        
if __name__=='__main__': rm_used_photo()

