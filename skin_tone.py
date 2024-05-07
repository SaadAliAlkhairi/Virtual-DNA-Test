import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_skin(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of the skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask using the skin color range
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    
    # Apply the mask to the original image
    skin_detected_image = cv2.bitwise_and(image, image, mask=skin_mask)
    
    return skin_detected_image, skin_mask

# Path to the image you want to analyze
image_path = 'father10.jpg'

# Detect skin tone in the image
skin_detected_image, skin_mask = detect_skin(image_path)

# Calculate the average color of the skin pixels
average_skin_color = cv2.mean(cv2.cvtColor(skin_detected_image, cv2.COLOR_BGR2RGB), mask=skin_mask)

# Display the original image, skin-detected image, and average skin color
original_image = cv2.imread(image_path)
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(skin_detected_image, cv2.COLOR_BGR2RGB))
plt.title('Skin Detected Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow([[average_skin_color[2] / 255, average_skin_color[1] / 255, average_skin_color[0] / 255]])
plt.title('Average Skin Color')
plt.axis('off')

plt.tight_layout()
plt.savefig('skin_analysis_result.png')  # Save the plot as an image file







# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_skin(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Convert the image from BGR to HSV color space
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define the lower and upper bounds of the skin color in HSV
#     lower_skin = np.array([0, 48, 80], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
#     # Create a mask using the skin color range
#     skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    
#     # Apply the mask to the original image
#     skin_detected_image = cv2.bitwise_and(image, image, mask=skin_mask)
    
#     return skin_detected_image, skin_mask

# # Path to the image you want to analyze
# image_path = 'father10.jpg'

# # Detect skin tone in the image
# skin_detected_image, skin_mask = detect_skin(image_path)

# # Calculate the average color of the skin pixels
# average_skin_color = cv2.mean(cv2.cvtColor(skin_detected_image, cv2.COLOR_BGR2RGB), mask=skin_mask)

# # Display the original image, skin-detected image, and average skin color
# original_image = cv2.imread(image_path)
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(skin_detected_image, cv2.COLOR_BGR2RGB))
# plt.title('Skin Detected Image')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow([[average_skin_color[2] / 255, average_skin_color[1] / 255, average_skin_color[0] / 255]])
# plt.title('Average Skin Color')
# plt.axis('off')

# plt.tight_layout()
# plt.show()





# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_skin(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Convert the image from BGR to HSV color space
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define the lower and upper bounds of the skin color in HSV
#     lower_skin = np.array([0, 48, 80], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
#     # Create a mask using the skin color range
#     skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    
#     # Apply the mask to the original image
#     skin_detected_image = cv2.bitwise_and(image, image, mask=skin_mask)
    
#     return skin_detected_image

# # Path to the image you want to analyze
# image_path = 'father10.jpg'

# # Detect skin tone in the image
# skin_detected_image = detect_skin(image_path)

# # Display the original image and the skin detected image
# # cv2.imshow('Original Image', cv2.imread(image_path))
# # cv2.imshow('Skin Detected Image', skin_detected_image)
# original_image = cv2.imread(image_path)
# # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
# # plt.title('Original Image')
# # plt.show()
# plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.savefig('original_image.png')  # Save the plot as an image file
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
