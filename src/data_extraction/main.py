import cv2

def display_image(file_path):

  image = cv2.imread(file_path)

  if image is None:
    print(f"Error: Unable to open image at'{file_path}'")
    return

  cv2.imshow('Image', image)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ =="__main__":

  file_path = '../../foto/image_20_0x_00175_22_6.png'
  display_image(file_path)
 
