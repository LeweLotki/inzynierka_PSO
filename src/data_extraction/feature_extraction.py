from numpy import (
    ndarray, float32, uint8
)

from numpy import (
    argmax, zeros_like, ones, any
)

from cv2 import (
    TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, KMEANS_RANDOM_CENTERS, CC_STAT_AREA, COLOR_BGR2GRAY
)

from cv2 import (
    calcHist, connectedComponents, dilate, erode, connectedComponentsWithStats, kmeans, cvtColor
)

from pandas import DataFrame

from os import path

class feature_extraction:
    
    num_clusters = 3
    kernel_size = 2
    csv_folder_path = r'..\\data\\'
    
    def __init__(self, image: ndarray, image_name: str):
        
        self.image = image
        self.image_name = image_name
        
        self.decrease_variety_of_intensity ()
        self.remove_adjacent_colors()
        self.closing_morphology()
        object_info = self.count_objects()
        self.write_csv(object_info=object_info)
           
    def decrease_variety_of_intensity (self):
        
        image = self.image
        image = cvtColor(image, COLOR_BGR2GRAY)
        
        pixels = image.reshape((-1, 3))
        pixels = float32(pixels)

        criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = kmeans(pixels, self.num_clusters, None, criteria, 10, KMEANS_RANDOM_CENTERS)

        centers = uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        self.image = segmented_image
    
    def remove_adjacent_colors(self):
        
        gray_image = self.image
        
        hist = calcHist([gray_image], [0], None, [256], [0, 256])

        highest_intensity = argmax(hist)
        hist[highest_intensity] = 0  
        middle_intensity = argmax(hist)
        hist[middle_intensity] = 0  
        lowest_intensity = argmax(hist)

        high_intensity_mask = (gray_image == highest_intensity)
        middle_intensity_mask = (gray_image == middle_intensity)
        lowest_intensity_mask = (gray_image == lowest_intensity)

        result = zeros_like(gray_image, dtype=uint8)
        result[high_intensity_mask] = 0
        result[middle_intensity_mask] = 255
        result[lowest_intensity_mask] = 255
                
        _, middle_labels = connectedComponents(middle_intensity_mask.astype(uint8))

        for label in range(1, middle_labels.max() + 1):
            
            current_component_mask = (middle_labels == label)
            dilated_mask = dilate(current_component_mask.astype(uint8), ones((3, 3), uint8))

            if any(dilated_mask & lowest_intensity_mask):
                result[current_component_mask] = 0

        self.image = result
    
    def closing_morphology(self):

        kernel = ones((self.kernel_size, self.kernel_size), uint8)
        
        binary_image = self.image
        
        dilated_image = dilate(binary_image, kernel, iterations=1)
        eroded_image = erode(dilated_image, kernel, iterations=1)
        corrected_image = dilate(eroded_image, kernel, iterations=1)

        self.image = corrected_image
    
    def count_objects(self) -> list:
        
        binary_image = self.image
        
        _, _, stats, centroids = connectedComponentsWithStats(binary_image)

        num_objects = len(stats) - 1  

        object_info = []

        for i in range(1, len(stats)):
            area = stats[i, CC_STAT_AREA]
            x, y, width, height, _ = stats[i]
            centroid = (int(centroids[i, 0]), int(centroids[i, 1]))
            
            top_left = (x, y)
            bottom_right = (x + width, y + height)

            object_info.append({
                'id': i,
                'area': area,
                'centroid': centroid,
                'rectangle': (top_left, bottom_right)
            })

        return object_info
    
    def write_csv(self, object_info: list):
            
        df = DataFrame(object_info)
        filename, _ = path.splitext(self.image_name)
        df.to_csv(self.csv_folder_path + filename + '.csv', index=False)
        
if __name__ == '__main__':
    
    file_path = r'..\..\foto\image_20_0x_00175_22_6.png'
    from cv2 import imread
    image = imread(file_path)
    feature_extraction(image=image, image_name='1')