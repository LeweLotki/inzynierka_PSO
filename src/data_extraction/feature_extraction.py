from numpy import (
    ndarray, max
)
from cv2 import (
    imread, imshow, waitKey, destroyAllWindows, 
    cvtColor, COLOR_BGR2GRAY, calcHist, GaussianBlur
)
from matplotlib.pyplot import (
    plot, title, xlabel, ylabel, show,
    subplot, axis, tight_layout, subplots
)

def display_histogram(img : ndarray) -> None: 
    
    histogram = calcHist([img], [0], None, [256], [0, 256])
    
    _, axs = subplots(1, 2, figsize=(10, 5))

    # Display the image in the first subplot
    axs[0].imshow(img / max(img), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Image')
    axs[0].axis('off')

    # Display the histogram in the second subplot
    axs[1].plot(histogram)
    axs[1].set_title('Image Histogram')
    axs[1].set_xlabel('Pixel Value')
    axs[1].set_ylabel('Frequency')

    tight_layout()


def feature_extraction(img : ndarray) -> ndarray: 

    # Step 1: Convert the image to black and white
    img = cvtColor(img, COLOR_BGR2GRAY)  
    
    display_histogram(img)
    
    img = GaussianBlur(img, (5, 5), 0)
    
    display_histogram(img)
    
    return img

if __name__ == '__main__':
    
    input_image = imread("..\\..\\foto\\image_20_0x_00175_22_6.png")

# Call the feature_extraction function
    result_image = feature_extraction(input_image)

    # Display the original and processed images for comparison
    # imshow('Original Image', input_image)
    # imshow('Processed Image', result_image)
    show()
    waitKey(0)
    destroyAllWindows()

