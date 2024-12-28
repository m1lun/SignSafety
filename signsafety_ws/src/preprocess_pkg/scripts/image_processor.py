#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import re
from PIL import Image as PILImage  # Rename the imported Image from PIL

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.publisher_ = self.create_publisher(Image, 'preprocessed_image', 10)
        self.bridge = CvBridge()
        # self.TOP_X_PADDING = 7
        # self.TOP_Y_PADDING = 3
        # self.BOTTOM_X_PADDING = 24
        # self.BOTTOM_Y_PADDING = 20
        self.blur_kernel = 12
        self.canny_low = 0
        self.canny_high = 195
        self.threshold_value = 127
        self.scale = 30
        image_path = r"/sim_ws/src/download1.png"
        image_path2 = r"/sim_ws/src/output.txt"
        self.get_image(image_path2)
        self.process_and_publish(image_path)

    def get_image(self, image_path):
        f = open(image_path, "r")
        pattern = re.compile(r"\d+")

        image_data = []

        for i in range(0, 129):
            f.readline()

        for i in range (0, 921600):
            image_data.append(int(pattern.findall(f.readline())[0]))
            
        data = image_data

        # Ensure the data length is divisible by 3
        if len(data) % 3 != 0:
            raise ValueError("Data length must be divisible by 3 to form RGB values.")

        # Group data into RGB tuples
        rgb_data = [(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]

        # Create an image with a single row of pixels
        width = 640
        height = 480
        image = PILImage.new("RGB", (width, height))

        # Set pixels in the image
        image.putdata(rgb_data)

        # Save the image
        output_path = "/sim_ws/src/download1.png"
        image.save(output_path)
        print(f"Image saved as {output_path}")

    def process_and_publish(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_copy = image.copy()
        if image is None:
            self.get_logger().error(f"Error: Unable to load image at {image_path}")
            return

        # Convert to HSV and mask red regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0,0,223])
        upper_bound = np.array([157,255,255])
        red_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Filter small regions

                # Draw the rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the ROI and store it
                cropped_rectangle = image[y:y+h, x:x+w]
                output_path = os.path.join(r'/sim_ws/src/trial', f"cropped_rectangle_{i + 1}.png")
                cv2.imwrite(output_path, cropped_rectangle)
                print(f"Saved cropped image {i + 1} to {output_path}")
                cropped_rectangle_rgb = cv2.cvtColor(cropped_rectangle, cv2.COLOR_BGRA2RGB)
                cropped_height, cropped_width, _ = cropped_rectangle_rgb.shape
                ros_image = self.bridge.cv2_to_imgmsg(cropped_rectangle_rgb, encoding='rgb8')
                ros_image.height = cropped_height
                ros_image.width = cropped_width
                ros_image.step = len(cropped_rectangle_rgb[0]) * cropped_rectangle_rgb.shape[2]  # Width * channels
                ros_image.data = cropped_rectangle_rgb.tobytes()
                ros_image.is_bigendian = 0  # Assuming little-endian
                ros_image.header.frame_id = str(10)
                ros_image.header.stamp = self.get_clock().now().to_msg()

                # Publish the cropped image
                self.publisher_.publish(ros_image)
                self.get_logger().info(f"Published image {i + 1} of {len(contours)}")


        # Ensure blur_kernel is odd
        blur_kernel = max(3, self.blur_kernel | 1)  # Make it odd and at least 3

        # Resize scale factor
        resize_factor = max(self.scale / 100, 0.01)

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image_copy, (blur_kernel, blur_kernel), 1.4)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, self.canny_low, self.canny_high)

        # Apply threshold
        _, threshold = cv2.threshold(edges, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            # Skip the first contour which is the whole image
            if i == 0:
                continue

            # Approximate the contour
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            # Highlight quadrilaterals
            if len(approx) == 4:  # Check if the contour has 4 vertices

                # Get the bounding box and crop
                x, y, w, h = cv2.boundingRect(approx)
                cropped_rectangle = image_copy[y:y+h, x:x+w]
                cropped_rectangle_rgb = cv2.cvtColor(cropped_rectangle, cv2.COLOR_BGRA2RGB)
                cropped_height, cropped_width, _ = cropped_rectangle_rgb.shape
                ros_image = self.bridge.cv2_to_imgmsg(cropped_rectangle_rgb, encoding='rgb8')
                ros_image.height = cropped_height
                ros_image.width = cropped_width
                ros_image.step = len(cropped_rectangle_rgb[0]) * cropped_rectangle_rgb.shape[2]  # Width * channels
                ros_image.data = cropped_rectangle_rgb.tobytes()
                ros_image.is_bigendian = 0  # Assuming little-endian
                ros_image.header.frame_id = str(10)
                ros_image.header.stamp = self.get_clock().now().to_msg()
                self.publisher_.publish(ros_image)
                self.get_logger().info(f"Published image {i + 1} of {len(contours)}")

                # # Save the cropped image to disk
                # output_path = os.path.join(r'/sim_ws/src', f"croppedssss_image_{i}.png")
                # cv2.imwrite(output_path, cropped_rectangle)
        
def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()