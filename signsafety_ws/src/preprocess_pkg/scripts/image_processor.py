#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import os

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.publisher_ = self.create_publisher(Image, 'preprocessed_image', 10)
        self.bridge = CvBridge()

        # # Load the image from a parameter
        # self.declare_parameter("image_path", "")
        # image_path = self.get_parameter("image_path").get_parameter_value().string_value

        # if not image_path:
        #     self.get_logger().error("No image path provided! Use --ros-args -p image_path:=<path>")
        #     return
        image_path = r"/sim_ws/src/download.png"
        
        self.process_and_publish(image_path)

    def process_and_publish(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            self.get_logger().error(f"Error: Unable to load image at {image_path}")
            return

        # Convert to HSV and mask red regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Highlight red regions
        red_highlighted = cv2.bitwise_and(image, image, mask=red_mask)
        height, width, _ = red_highlighted.shape

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Filter small regions
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Define cropped images (example of cropping)
        cropped_images = []
        crop_size = (100, 100)  # Example size of the cropped image (adjust as needed)
        for y in range(0, height, crop_size[1]):
            for x in range(0, width, crop_size[0]):
                cropped_image = image[y:y+crop_size[1], x:x+crop_size[0]]
                cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cropped_images.append(cropped_gray)

        # Get the directory of the original image for saving the output
        output_dir = os.path.dirname(image_path)

        # Save the image with bounding boxes
        output_image_path = os.path.join(output_dir, "image_with_bounding_boxes.png")
        cv2.imwrite(output_image_path, image)

        # Save the cropped grayscale images
        for i, cropped_gray in enumerate(cropped_images):
            cropped_image_path = os.path.join(output_dir, f"cropped_gray_{i+1}.png")
            cv2.imwrite(cropped_image_path, cropped_gray)

        # Optionally, print out where the files were saved
        print(f"Saved images to {output_dir}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
