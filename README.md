import cv2

def enhance_image(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection to enhance edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply bilateral filter for noise reduction while preserving edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply histogram equalization for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Save the enhanced image
    cv2.imwrite('enhanced_image.jpg', enhanced)

# Example usage
enhance_image('input_image.jpg')
