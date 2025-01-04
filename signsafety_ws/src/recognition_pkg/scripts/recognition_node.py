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

        self.publisher_test = self.create_publisher(Image, 'preprocessed_image', 10)
        self.publisher_sign = self.create_publisher(String, '/recognized_sign', 10)
        self.publisher_start = self.create_publisher(String, '/car_start', 10)

        self.bridge = CvBridge()

        self.modelnum = 2
        self.model = tf.keras.models.load_model(f"models/model{self.modelnum}")
        self.input_shape = self.model.input_shape[1:3] # this is (H, W)
        self.supported_indices = [0, 1, 2, 3, 4, 5, 7, 8, 13, 14]
        self.labels = ['SPEED_LIMIT;20', 'SPEED_LIMIT;30', 'SPEED_LIMIT;50', 'SPEED_LIMIT;60', 'SPEED_LIMIT;70', 'SPEED_LIMIT;80', 'SPEED_LIMIT;100', 'SPEED_LIMIT;120', 'YIELD', 'STOP'] # Add more once these are working

        self.prob_threshold = 0.98

        self.get_logger().info("Recognition Node Initialized")

        start_msg = String()
        start_msg.data = "START"
        self.publisher_start.publish(start_msg)

    def image_callback(self, msg):
        try:
       
            # Convert ROS Image message to numpy array
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            
            # Resize and normalize the image for model
            img_resized = cv2.resize(cv_image, self.input_shape)
            img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Perform sign recognition via model1
            predictions = self.model.predict(img_array)
            max_prob = np.max(predictions)
            recognized_label_index = np.argmax(predictions, axis=1)[0]

            if max_prob < self.prob_threshold:
                recognized_sign = "unknown"
                self.get_logger().info(f"Index is {recognized_label_index} with (low) prob {max_prob}")
                return
                
            if recognized_label_index not in self.supported_indices:
                self.get_logger().warning(f"Index {recognized_label_index} is out of bounds or unsupported. Sign not recognized.")
                recognized_sign = "unknown"
                return

            recognized_sign = self.labels[self.supported_indices.index(recognized_label_index)]
            distance = float(msg.header.frame_id) if msg.header.frame_id else 0.0
            self.get_logger().info(f"Index is {recognized_label_index} with prob {max_prob}")
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
