#!/usr/bin/env python3
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
import tensorflow as tf
import numpy as np
import cv2
from rclpy.node import Node
import rclpy
import time
from cv_bridge import CvBridge

class RecognitionNode(Node):
    def __init__(self):
        super().__init__('recognition_node')

        # Initialize subscribers and publishers
        self.subscription = self.create_subscription(
            Image,
            '/preprocessed_image',
            self.image_callback,
            10
        )

        self.publisher_test = self.create_publisher(Image, 'preprocessed_image', 10);
        self.publisher_sign = self.create_publisher(String, '/recognized_sign', 10)
        
        # Load your ML model (adjust the path to your model)
        self.model = tf.keras.models.load_model('/home/milun/SignSafety/signsafety_ws/models/model1')
        self.input_shape = self.model.input_shape[1:3] # this is (H, W)
        self.labels = [str(i) for i in range(self.model.output_shape[1])]  # Class IDs as labels
        # self.labels = ['Stop', 'Speed Limit'] # Add more once these are working EVENTUALLY WE USE THIS

        self.get_logger().info("Recognition Node Initialized")

        time.sleep(5)  # Wait 5 seconds before publishing

        image_msg = Image()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "12"
        image_msg.height = 480  # Example height
        image_msg.width = 640   # Example width
        image_msg.encoding = "rgb8"  # Example encoding
        image_msg.is_bigendian = False
        image_msg.step = 640 * 3  # Example step (width * number of channels)
        image_msg.data = [0] * (480 * 640 * 3)  # Dummy black image


        # Publish the image
        self.publisher_test.publish(image_msg)
        self.get_logger().info("Published preprocessed image to /preprocessed_image")

    def image_callback(self, msg):
        try:
       
            # Convert ROS Image message to numpy array
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            
            # Resize and normalize the image for model input
            img_resized = cv2.resize(cv_image, self.input_shape)
            img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Perform sign recognition
            predictions = self.model.predict(img_array)
            recognized_label_index = np.argmax(predictions, axis=1)[0]
            self.get_logger().info(f"Index is {recognized_label_index}")
            recognized_sign = self.labels[recognized_label_index]
            
            # extract distance
            distance = float(msg.header.frame_id) if msg.header.frame_id else 0.0

            # Publish recognized sign and relay the distance
            self.publish_sign(recognized_sign, distance)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def publish_sign(self, recognized_sign, distance):
        """
        Publish the recognized sign as a String message.
        """
        msg = String()
        msg.data = f"{recognized_sign};{distance}"
        self.publisher_sign.publish(msg)
        self.get_logger().info(msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = RecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Recognition Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
