"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(Node):
    def __init__(self):
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            1)
        self.pose_sub_

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        self.scan_sub_

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10)

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.NEIGHBOR_RADIUS = 10.0
        self.GOAL_DISTANCE = 10.0
        self.grid_size = 20  # Grid will cover 20 x 200
        self.grid_origin_x = -10  
        self.grid_origin_y = -10
        self.x_min, self.x_max = -10, 10
        self.y_min, self.y_max = -10, 10
        self.lookahead_distance = 1  # Adjust based on testing
        
        # Create a numpy array to represent the occupancy grid
        self.occupancy_grid = np.zeros((self.grid_size),(self.grid_size))
        self.load_waypoints()

    def load_waypoints(self):
        # Load waypoints from a .csv file
        # The CSV file should have columns: Position, Orientation, Angle (no headers)
        waypoint_file = '/sim_ws/src/lab-7-slam-crashout/pure_pursuit/scripts/pose_data.csv'  # Adjust the path to your file
        try:
            with open(waypoint_file, newline='') as csvfile:
                waypoint_reader = csv.reader(csvfile)
                self.waypoints = []
                
                for row in waypoint_reader:
                    if len(row) >= 3:  # Ensure there are enough columns
                        # Extract the position (first column) and convert it to a tuple (x, y)
                        position_str = row[0]  # Position is in the form "(x, y, z)"
                        position_str = position_str.strip('()')  # Remove parentheses
                        position_values = position_str.split(',')  # Split the string by commas
                        if len(position_values) >= 2:
                            x = float(position_values[0].strip())  # Convert x to float
                            y = float(position_values[1].strip())  # Convert y to float
                            self.waypoints.append((x, y))  # Append (x, y) tuple to the waypoints list
                            
                            # Log the x and y values
            #                 self.get_logger().info(f'Loaded waypoint: x={x}, y={y}')

            # self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints.')
            self.visualize_waypoints()

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # Clear the grid
        self.occupancy_grid.fill(0)
        
        # Iterate through the laser scan data
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        for i, range_reading in enumerate(scan_msg.ranges):
            if range_reading < scan_msg.range_max:
                # Calculate the angle of the laser ray
                angle = angle_min + i * angle_increment

                # Convert polar to cartesian coordinates (relative to the car)
                x = range_reading * math.cos(angle)
                y = range_reading * math.sin(angle)

                # Convert to grid coordinates
                grid_x = (x - self.grid_origin_x)
                grid_y = (y - self.grid_origin_y)

                # Mark the grid cell as occupied
                if 0 <= grid_x < self.occupancy_grid.shape[0] and 0 <= grid_y < self.occupancy_grid.shape[1]:
                    self.occupancy_grid[grid_x, grid_y] = 1

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """

        tree = [Node()] 

        x_init = Node()
        x_init.x = pose_msg.pose.position.x
        x_init.y = pose_msg.pose.position.y
        tree.append(x_init)
        
        # Use lookahead distance to get goal_node
        self.goal_node = self.find_closest_waypoint(car_x, car_y)

        # Assume RRT
        while True:
            x_rand = self.sample() # Random node
            x_nearest = self.nearest(tree, x_rand) # Nearest Node already in graph
            x_new = self.steer(x_nearest, x_rand) # Point to steer in between
        
            if(self.check_collision(x_new, x_nearest)): # Ensure path from existing to new is free
                continue
            
            x_new.parent = x_nearest # Add edge
            tree.append(x_new) # Add Vertex
            
            if self.is_goal(x_new, self.goal_node): # If our current_node actually reaches the goal
                # Give steering message towards x_new
                path = self.find_path(tree, x_new)
                dx = path[0].x - x_init.x
                dy = path[0].y - x_init.y
                goal_distance = np.sqrt(dx**2 + dy**2)
                curvature = 2 * dy / (goal_distance**2)
                drive_msg = AckermannDriveStamped()
                drive_msg.drive.steering_angle = curvature 
                drive_msg.drive.speed = 1.0  # Set to a suitable speed
                self.drive_pub.publish(drive_msg)
                break
            
    def find_closest_waypoint(self, car_x, car_y):
        # Find the closest waypoint that's at least `lookahead_distance` away
        for i, (wp_x, wp_y) in enumerate(self.waypoints[0:]):
            distance = np.sqrt((wp_x - car_x)**2 + (wp_y - car_y)**2)
            if distance >= self.lookahead_distance:
                return i
        return len(self.waypoints) - 1  # Return last waypoint if none found

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """        
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        
        return (x, y)


    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        nearest_node = None
        min_distance = float('inf')
        
        for node in tree:
            distance = math.sqrt((node.x - sampled_point[0])**2 + (node.y - sampled_point[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        step_size = 0.1
        direction = np.array(sampled_point) - np.array([nearest_node.x, nearest_node.y])
        distance = np.linalg.norm(direction)
        if distance == 0:
            return nearest_node  # No movement
        
        # Normalize direction and scale by step_size
        direction = (direction / distance) * min(step_size, distance)
        new_x = nearest_node.x + direction[0]
        new_y = nearest_node.y + direction[1]
                
        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """

        path = self.find_path(nearest_node, new_node)

        # Check each node along the path and if occupied, return true
        for node in path:
            if OccupancyGrid[node.x, node.y]:
                return True

        return False

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """

        # Check if the pythagorean distance is less than the paramter GOAL_DISTANCE
        return math.sqrt((latest_added_node.x - goal_x)**2 + (latest_added_node.y - goal_y)**2) <= self.GOAL_DISTANCE

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        curr_node = latest_added_node

        # Keep back tracking until we find the root
        while curr_node is not None and not curr_node.is_root:
            path.append(curr_node)
            curr_node = tree[curr_node.parent]

        path.reverse() # Reverse since we want the root at the start

        return path


    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """

        # Since parent node has a calculated cost just add that to the new line cost
        if tree[node.parent] is None or tree[node.parent].is_root:
            return 0.0
        else:
            return tree[node.parent.cost] + self.line_cost(node, tree[node.parent])

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """

        # cost calculated using pythagorean distance between the points
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """

        neighborhood = []
        for other in tree:
            if self.line_cost(node, other) <= self.NEIGHBOR_RADIUS and node is not other:
                neighborhood.append(other)

        return neighborhood
    def visualize_waypoints(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()  # Set current timestamp
        marker.type = Marker.POINTS
        marker.ns = "basic_shapes"
        marker.id = 0

        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        # marker.lifetime = Duration(sec=0, nanosec=0)  # Lifetime of zero (persistent marker)

        for i, (x, y) in enumerate(self.waypoints):
            p = Point()
            p.x = x
            p.y = y
            marker.points.append(p)
            if i == self.current_waypoint_index:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            marker.color.a = 1.0

        self.waypoint_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()