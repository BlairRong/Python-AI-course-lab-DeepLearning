"""
Image Processing Lab with OpenCV and NumPy

2.26 Lab: basic image operations: 
Load and Inspect an Image;
Access and Modify Pixels; 
Modify a Region of the image.

"""

import cv2
import numpy as np


# Part 1: Load and Inspect an Image 加载和检查图像

# Task 1: Load an image from disk using OpenCV
image_path = 'input.jpg'
img = cv2.imread(image_path)

# Task 2: Check that the image loaded correctly (handle errors if not)
if img is None:
    print(f"Error: Could not load image from {image_path}. Please check the file.")
    exit()
    
# Task 3: Print the image shape
print("Image shape (height, width, channels):", img.shape)
# Image shape (height, width, channels):(1133, 1133, 3)

# Questions：
#   Q: What do the three values in the shape represent?
#   A: (1133, 1133, 3) They represent (height, width, number of color channels).

#	Q: Which value corresponds to height? width? channels?
#   A: First value 1133 = height, second 1133 = width, third 3 = channels.

#	Q: What is the data type of the image array? 图像数组的数据类型
print("Data type of image array:", img.dtype)
#   A: The data type is uint8 (unsigned 8-bit integer, range 0-255).






# Part 2: Access and Modify Pixels 访问和修改像素

# Task 1: Access a single pixel(BGR values) at a specific coordinate 访问特定坐标处的单个像素 (e.g. row 100, column 150)
row, col = 100, 150
pixel = img[row, col]

# Task 2: Print its values
print(f"Original pixel at ({row},{col}): BGR = {pixel}")
# Original pixel at (100,150): BGR = [136 136 136]

# Task 3: Modify that pixel to a new color 将该像素修改为新颜色 (e.g., bright cyan 亮青色: B=255, G=255, R=0)
new_color = (255, 255, 0) #OpenCV uses BGR order
img[row, col] = new_color

# Task 4: Display the updated image
cv2.imshow('Modified Pixel', img)
cv2.imwrite('modified_pixel.jpg', img)
print("modified_pixel save to modified_pixel.jpg")

#Questions：
#	Q: In what order are color values stored in OpenCV?
#   A: They are stored in BGR order (Blue, Green, Red), not RGB.

#	Q: What happens when you set all values to maximum?
#   A: Setting all three channels to 255 results in white (in BGR, that's also white because full intensity of all colors).

#	Q: What happens when you set all values to zero?
#   A: Setting all channels to 0 results in black.





# Part 3: Modify a Region of the Image 修改图像区域

# Task 1: Select a rectangular region of the image using slicing 使用切片功能选择图像中的一个矩形区域
# Define a rectangular region: 行rows 200 to 300, 列columns 300 to 400
region = img[200:300, 300:400]

# Task 2: Change the entire region to a single color 将整个区域更改为单一颜色 (e.g. green in BGR: 0, 255, 0)
region[:] = (0, 255, 0)  # This modifies the original image because region is a view
# Alternatively, could assign directly: img[200:300, 300:400] = (0, 255, 0)

# Task 3: Display the result
cv2.imshow('Modified Region', img)
cv2.imwrite('modified_region.jpg', img)
print("modified region save to modified_region.jpg")

#Questions：
#	Q: How does slicing work in NumPy for images? NumPy 中的图像切片功能是如何工作的？
#   A: Slicing uses the syntax语法： img[start_row:end_row, start_col:end_col] to extract提取 a sub-array 子数组.
#      The slice returns a view of the original data (if no stride tricks) so modifying it affects the original data.

#	Q: What do the row and column ranges represent? 行和列范围分别代表什么？
#   A: The row range [start_row:end_row) selects rows from start_row (inclusive) to end_row (exclusive)不包含.
#      Similarly, column range selects columns. 
#      So the region区域 is a rectangle矩形 of size 长为(end_row - start_row) by(x) 宽为(end_col - start_col).
