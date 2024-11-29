#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
import math
import logging

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

# Path to log file
readings_log_file = os.path.join('/home/anuhaad/sim_ws', 'lidar_readings.log')

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.subscription = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10)
        
        # Publish to drive
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10)

        # Angle fromr reading
        self.max_angle = 0.0
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.scan_ranges_max = 0.0
        self.car_width = 0.5
        self.last_gap = 0.0
        self.last_best_points = [540, 541]
        self.bubble_size = 16

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        f = open(readings_log_file, 'a')  # 'w' mode overwrites the file
        f.write('[' + ', '.join(map(str, ranges)) + ']\n')  # Add brackets around the list
        f.close()

        proc_ranges = np.array(ranges)
        proc_ranges[proc_ranges > 7.0] = 7.0
        proc_ranges[np.isnan(proc_ranges)] = 0.0  # Correctly handle NaN values
        
        # Calculate angle limits in radians (±75 degrees = ±75 * π / 180 radians)
        fov_limit = 75 * math.pi / 180.0
        start_angle = -fov_limit 
        end_angle = fov_limit    
        
        # Compute the indices based on angle_min and angle_increment
        start_index = max(0, int((start_angle - self.angle_min) / self.angle_increment))
        end_index = min(len(ranges), int((end_angle - self.angle_min) / self.angle_increment))

        # Restrict the LiDAR data to the range between start_index and end_index
        proc_ranges[:start_index] = 0
        proc_ranges[end_index:] = 0
        
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """

        max_gap = 0
        start_i = 0
        end_i = 0
        best_indices = [239, 840]
        max_range = 0


        for i in range(0, len(free_space_ranges)):
            
            # Find the start of the gap
            if free_space_ranges[i] > 0.0:
                start_i = i

                # Find the end of the gap
                for j in range(i, len(free_space_ranges)):
                    if free_space_ranges[j] == 0.0:
                        end_i = j
                        i = j
                        break
                gap = end_i - start_i

                # Check if the current gap is the largest gap
                if gap > max_gap:
                    max_gap = gap
                    max_range = max(free_space_ranges[start_i:end_i])
                    best_indices = [start_i, end_i]
                    
        return best_indices
        
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # Find the index of the maximum distance within the gap
        best_point = start_i + np.argmax(ranges[start_i:end_i+1])
        another_best_point = 0
        for i in range(best_point, len(ranges)):
            if ranges[i] != ranges[best_point]:
                another_best_point = i
                break

        return (another_best_point + best_point) // 2
        
    def find_disparity_and_extend(self, ranges):
        """
        Finds disparities in the LIDAR data and extends them based on the car's width.
        """
        extended_ranges = np.copy(ranges)

        # Loop through ranges and find disparities
        for i in range(1, len(ranges)):
            if (extended_ranges[i] != 0.0 and extended_ranges[i - 1] != 0.0):
                disparity = extended_ranges[i] - extended_ranges[i - 1]
                if abs(disparity) > 1.0:  # Threshold for detecting obstacles
                    # Extend the disparity by half the car's width
                    if ranges[i - 1] > ranges[i]:
                        extend_range = int(self.car_width / 2 / self.angle_increment / ranges[i])
                    else:
                        extend_range = int(self.car_width / 2 / self.angle_increment / ranges[i - 1])
                    start = max(0, i - extend_range)
                    end = min(len(ranges), i + extend_range)
                    extended_ranges[start:end] = min(ranges[i - 1], ranges[i])
                    # extended_ranges[i - 1] = 0.0
                    i += extend_range

        return extended_ranges
 
    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        self.max_angle = data.angle_max
        self.angle_min = data.angle_min
        self.angle_increment = data.angle_increment
        self.scan_ranges_max = data.range_max 

        # Preprocess
        proc_ranges = self.preprocess_lidar(ranges)

        f = open(readings_log_file, 'a')
        f.write('[' + ', '.join(map(str, proc_ranges)) + ']\n')  # Add brackets around the list
        f.close()

        # Find disparities and extend them based on car width
        extended_ranges = self.find_disparity_and_extend(proc_ranges)

        # Find the largest gap
        start_i, end_i = self.find_max_gap(extended_ranges)

        # Find the best point in the gap
        best_point = self.find_best_point(start_i, end_i, extended_ranges)
        
        # Print
        angle = data.angle_min + best_point * data.angle_increment
        best_range = extended_ranges[best_point]
        self.get_logger().info(f'Best point index: {best_point}, Angle: {angle:.1f}, Range: {best_range:.10f}')

        speed = 0.5

        # Publish 
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = angle
        self.publisher_.publish(msg)


def main(args=None):
    # Overwrites the file
    f = open(readings_log_file, 'w')
    f.close()

    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()