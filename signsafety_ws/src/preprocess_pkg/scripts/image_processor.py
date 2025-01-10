#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import re
import time
from PIL import Image as PILImage  # Rename the imported Image from PIL

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor')

        # Topic for images
        rgb_topic = '/camera/camera/color/image_raw'
        depth_topic = '/camera/camera/depth/image_rect_raw'


        self.publisher_ = self.create_publisher(Image, 'preprocessed_image', 10)
        self.publisher_test = self.create_publisher(Image, 'preprocessed_image', 10) 
   
        # Setup for the image cropping
        self.min_dist = 480 / 8
        self.param1 = 100
        self.param2 = 30
        self.min_radius = 1
        self.max_radius = 30

        # Subscribe to image topic
        self.rgb_subscriber = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_subcriber = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
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
        image_path = r"test/test.png"
        image_path2 = r"src/preprocess_pkg/output.txt"

        time.sleep(10)
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error(f"Failed to load image from {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.process_and_publish(image)

        # time.sleep(1)
        # self.get_image(image_path2)
        # self.process_and_publish(image_path)

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
        output_path = "test/test.png"
        image.save(output_path)
        print(f"New image saved as {output_path}")

    def process_and_publish(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        original_img = image.copy()
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
                output_path = os.path.join(r'test/pre_output', f"cropped_rectangle_{i + 1}.png")
                cv2.imwrite(output_path, cropped_rectangle)
                #print(f"Saved cropped image {i + 1} to {output_path}")
                cropped_rectangle_rgb = cv2.cvtColor(cropped_rectangle, cv2.COLOR_BGR2RGB)
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
                #self.get_logger().info(f"Published image {i + 1} of {len(contours)}")
                cv2.imshow("CROPPED RECTANGLE", cropped_rectangle)

        # Resize the image
        resized_img = cv2.resize(original_img, (0, 0), fx=1.4, fy=1.4)

        # Convert to grayscale
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))

        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        # Display and process the results
        output = resized_img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for idx, pt in enumerate(circles[0, :]):
                a, b, r = pt[0], pt[1], pt[2]
                cv2.circle(output, (a, b), r, (0, 255, 0), 2)
                cv2.circle(output, (a, b), 1, (0, 0, 255), 3)
                
                # Crop the circle
                x, y, w, h = a - r, b - r, 2 * r, 2 * r
                cropped_circle = resized_img[y:y+h, x:x+w]
                
                # Create a mask for the circle
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (r, r), r, 255, -1)
                
                # Apply the mask to the cropped circle
                cropped_circle_masked = cv2.bitwise_and(cropped_circle, cropped_circle, mask=mask)
                
                # Convert to ROS image and publish
                cropped_circle_rgb = cv2.cvtColor(cropped_circle_masked, cv2.COLOR_BGR2RGB)
                cropped_height, cropped_width, _ = cropped_circle_rgb.shape
                ros_image = self.bridge.cv2_to_imgmsg(cropped_circle_rgb, encoding='rgb8')
                ros_image.height = cropped_height
                ros_image.width = cropped_width
                ros_image.step = cropped_width * 3  # Width * channels (RGB has 3 channels)
                ros_image.data = cropped_circle_rgb.tobytes()
                ros_image.is_bigendian = 0  # Assuming little-endian
                ros_image.header.frame_id = str(10)
                ros_image.header.stamp = self.get_clock().now().to_msg()
                self.publisher_.publish(ros_image)
                #self.get_logger().info(f"Published circle image {idx + 1} of {len(circles[0])}")

                # Save the cropped circle image for reference
                cv2.imwrite(f'test/pre_output/Cropped_Circle_{idx}.png', cropped_circle_masked)
                cv2.imshow("CROPPED CIRCLE", cropped_circle_masked)

    def rgb_callback(self, data):
        #self.get_logger().warning("Receiving RGB data")
        current_frame = self.bridge.imgmsg_to_cv2(data)
        # Convert to RGB since cv2 default is BGR when reading
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("RGB", current_frame)
        self.process_and_publish(current_frame)
        time.sleep(0.1)  # Wait 51seconds before publishing
        cv2.waitKey(1)

    def depth_callback(self, data):
        #self.get_logger().warning("Receiving depth data")
        current_frame = self.bridge.imgmsg_to_cv2(data)
        cv2.imshow("DEPTH", current_frame)
        time.sleep(0.1)  # Wait 1 seconds before publishing
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
