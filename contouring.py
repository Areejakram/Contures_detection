import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
path ="C:\\Users\\hp\\Google Drive\\Fiverr Work\\2022\\15. Teaching OpenCV to Client\\Pics+scripts\\Pictures"
path ="C:\\Users\\hp\\Google Drive\\Fiverr Work\\2022\\15. Teaching OpenCV to Client\\Pics+scripts\\Pictures"
# read image
image = cv.imread(path + "\\icons01.png")

# plot using matplotlib library
plt.figure(1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
# convert to gray before moving countoring
grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# now find all contours in the image
contours, hierarchy = cv.findContours(grayImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
lenghthOfContours = len(contours)
print("the total number of contours in the image:" + str(lenghthOfContours))
# make copy of original image
imgCopy = image.copy()

# draw all contours
cv.drawContours(imgCopy, contours, -1, (0,255,0), 10)


plt.figure(figsize= [10,10])
plt.imshow(imgCopy)
plt.title("Image with Contours")
plt.axis("off")
# read an image
img2 = cv.imread(path + "\\tree.jpg")

# display 
plt.figure(1)
plt.imshow(img2)
plt.title("original image")
plt.axis("off")
# make copy 
copyImg = np.copy(img2)
print(copyImg.shape)

# convert it to gray
imgGray = cv.cvtColor(copyImg, cv.COLOR_BGR2GRAY)
print(imgGray.shape)

# display 
plt.figure(2)
plt.imshow(imgGray, cmap="gray")
plt.title("grayscale image")
plt.axis("off")
invertGray = cv.bitwise_not(imgGray)

# display 
plt.figure(3)
plt.imshow(invertGray, cmap="gray")
plt.title("Inverted grayscale image")
plt.axis("off")
# now find the contours
contours, hierarchy = cv.findContours(imgGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# draw all contours
cv.drawContours(copyImg, contours, -1, (255, 0, 0), 5)

# Display the result  
plt.imshow(copyImg)
plt.title("Contours Detected")
plt.axis("off")
ret, binaryThresh = cv.threshold(invertGray, 10, 255, cv.THRESH_BINARY)
print(ret)

# Display the result 
plt.imshow(binaryThresh, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
# make another copy
copyImg2 = img2.copy()

# now find the contours
contours, hierarchy = cv.findContours(binaryThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# draw all contours
cv.drawContours(copyImg2, contours, -1, (255, 0, 255), 5)

# Plot both of the resuts for comparison
plt.figure(figsize=[10, 10])
plt.subplot(121)
plt.imshow(copyImg)
plt.title("Without Thresholding")
plt.axis('off')
plt.subplot(122)
plt.imshow(copyImg2)
plt.title("With Threshloding")
plt.axis('off')
# Read the image
image3 = cv.imread(path + '\\chess.jpg') 

# Display the image
plt.figure(figsize=[10,10])
plt.imshow(image3)
plt.title("Original Image")
plt.axis("off")
cp = image3.copy()

# Blur the image to remove noise
blurred_image = cv.GaussianBlur(cp,(5,5),0)

# Apply canny edge detection
edges = cv.Canny(blurred_image, 100, 160)

# Display the resultant binary image of edges
plt.figure(figsize=[10,10])
plt.imshow(edges,cmap='gray')
plt.title("Edges Image")
plt.axis("off")
# Detect the contour using the using the edges
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw the contours
image3_copy = image3.copy()

cv.drawContours(image3_copy, contours, -1, (0, 255, 0), 2)

# Display the drawn contours
plt.figure(figsize=[10,10])
plt.imshow(image3_copy)
plt.title("Contours Detected")
plt.axis("off")
image3_copy2 = image3.copy()

# Remove noise from the image
blurred = cv.GaussianBlur(image3_copy2,(3,3),0)

# Convert the image to gray-scale
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

# Perform adaptive thresholding 
binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 5)

# Detect and Draw contours
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(image3_copy2, contours, -1, (0, 255, 0), 2)

# Plotting both results for comparison
plt.figure(figsize=[18,18])
plt.subplot(121)
plt.imshow(image3_copy2)
plt.title("Using Adaptive Thresholding")
plt.axis('off')
plt.subplot(122)
plt.imshow(image3_copy)
plt.title("Using Edge Detection")
plt.axis('off')

image1_copy = image.copy()

# Find all contours in the image.
contours, hierarchy = cv.findContours(grayImage, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# Select a contour
# index = 2
# contour_selected = contours[index]

# Draw the selected contour
cv.drawContours(image1_copy, contours, 2, (0,255,0), 6)

# Display the result
plt.figure(figsize=[10,10])
plt.imshow(image1_copy)
plt.axis("off")
plt.title('Selected Contour: ' + str(index))
image1_copy = image.copy()

# Create a figure object for displaying the images
plt.figure(figsize=[20,10])

# Convert to grayscale.
imageGray = cv.cvtColor(image1_copy, cv.COLOR_BGR2GRAY)

# Find all contours in the image
contours, hierarchy = cv.findContours(imageGray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# Loop over the contours
for i,cont in enumerate(contours):
        
        # Draw the ith contour
        image1_copy = cv.drawContours(image.copy(), cont, -1, (0,255,0), 10)

        # Add a subplot to the figure
        plt.subplot(3, 2, i+1)  

        # Turn off the axis
        plt.axis("off");plt.title('contour ' +str(i))

        # Display the image in the subplot
        plt.imshow(image1_copy)
