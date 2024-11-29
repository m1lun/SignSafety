#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>

#define FOV_RANGE 75                 // Points outside this range (in degrees) will be disregarded
#define LONGEST_RANGE_THRESHOLD 7.0  // Points further than this threshold will be truncated to the threshold

// Log file path for LIDAR readings
const std::string readings_log_file = "/home/anuhaad/sim_ws/lidar_readings.log";

class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node") {
        // Define LIDAR scan and drive topics
        auto lidarscan_topic = "/scan";
        auto drive_topic = "/drive";

        // Subscribe to LIDAR scan messages
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidarscan_topic, 10, std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1));

        // Publisher for driving messages
        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);

        // Initialize member variables
        max_angle_ = 0.0;
        angle_min_ = 0.0;
        angle_increment_ = 0.0;
        scan_ranges_max_ = 0.0;
        car_width_ = 0.5;
    }

private:
    // Callback function for LIDAR data processing
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr data) {
        std::vector<float> ranges = data->ranges;
        max_angle_ = data->angle_max;
        angle_min_ = data->angle_min;
        angle_increment_ = data->angle_increment;
        scan_ranges_max_ = data->range_max;

        // Preprocess the LIDAR data
        std::vector<float> proc_ranges = preprocess_lidar(ranges);

        // Log processed LIDAR data
        log_readings(proc_ranges);

        // Find and extend disparities in LIDAR readings
        std::vector<float> extended_ranges = find_disparity_and_extend(proc_ranges);

        // Find the largest gap in extended LIDAR data
        auto [start_i, end_i] = find_max_gap(extended_ranges);

        // Identify the best point in the largest gap
        int best_point = find_best_point(start_i, end_i, extended_ranges);

        // Calculate angle to steer towards the best point
        float angle = angle_min_ + best_point * angle_increment_;
        float best_range = extended_ranges[best_point];

        // Log the best point details
        RCLCPP_INFO(this->get_logger(), "Best point index: %d, Angle: %.1f, Range: %.10f", best_point, angle, best_range);

        // Publish driving command with the calculated angle and speed
        publish_drive_message(angle, 0.5);
    }

    // Preprocess LIDAR data to remove invalid or overly large readings
    std::vector<float> preprocess_lidar(const std::vector<float>& ranges) {
        std::vector<float> proc_ranges(ranges);

        // Set field of view limit in radians (±75 degrees)
        float fov_limit = FOV_RANGE * M_PI / 180.0;

        // Compute start and end indices based on field of view
        int start_index = std::max(0, static_cast<int>((-fov_limit - angle_min_) / angle_increment_));
        int end_index = std::min(static_cast<int>(ranges.size()), static_cast<int>((fov_limit - angle_min_) / angle_increment_));

        // Filter LIDAR data within the specified field of view and valid ranges
        for (size_t i = 0; i < proc_ranges.size(); ++i) {
            if (proc_ranges[i] > LONGEST_RANGE_THRESHOLD) proc_ranges[i] = LONGEST_RANGE_THRESHOLD;
            if (std::isnan(proc_ranges[i])) proc_ranges[i] = 0.0;
            if (i < start_index || i > end_index) proc_ranges[i] = 0.0;
        }
        return proc_ranges;
    }

    // Find the largest gap in LIDAR data
    std::pair<int, int> find_max_gap(const std::vector<float>& free_space_ranges) {
        int max_gap = 0, start_i = 0, end_i = 0;
        std::pair<int, int> best_indices = {239, 840};

        for (size_t i = 0; i < free_space_ranges.size(); ++i) {
            if (free_space_ranges[i] > 0.0) {
                int j = i;
                while (j < free_space_ranges.size() && free_space_ranges[j] > 0.0) j++;
                int gap = j - i;
                if (gap > max_gap) {
                    max_gap = gap;
                    best_indices = {i, j - 1};
                }
                i = j;
            }
        }
        return best_indices;
    }

    // Identify the best point to target in the identified gap
    int find_best_point(int start_i, int end_i, const std::vector<float>& ranges) {
        auto max_it = std::max_element(ranges.begin() + start_i, ranges.begin() + end_i + 1);
        int best_point = std::distance(ranges.begin(), max_it);
        int another_best_point = best_point;
        
        // Further refine the best point selection
        for (size_t i = best_point; i < ranges.size(); ++i) {
            if (ranges[i] != ranges[best_point]) {
                another_best_point = i;
                break;
            }
        }
        return (another_best_point + best_point) / 2;
    }

    // Extend disparities in the LIDAR data for obstacle avoidance
    std::vector<float> find_disparity_and_extend(const std::vector<float>& ranges) {
        std::vector<float> extended_ranges = ranges;

        for (size_t i = 1; i < ranges.size(); ++i) {
            if (extended_ranges[i] != 0.0 && extended_ranges[i - 1] != 0.0) {
                float disparity = std::abs(extended_ranges[i] - extended_ranges[i - 1]);
                if (disparity > 1.0) {
                    int extend_range = car_width_ / 2 / angle_increment_ / (extended_ranges[i - 1] > extended_ranges[i] ? extended_ranges[i] : extended_ranges[i - 1]);
                    int start = std::max(0, static_cast<int>(i) - extend_range);
                    int end = std::min(static_cast<int>(ranges.size()), static_cast<int>(i) + extend_range);
                    std::fill(extended_ranges.begin() + start, extended_ranges.begin() + end, std::min(extended_ranges[i - 1], extended_ranges[i]));
                }
            }
        }
        return extended_ranges;
    }

    // Log LIDAR readings to file for debugging or analysis
    void log_readings(const std::vector<float>& readings) {
        std::ofstream log_file(readings_log_file, std::ios::app);
        if (log_file.is_open()) {
            log_file << "[";
            for (size_t i = 0; i < readings.size(); ++i) {
                log_file << readings[i];
                if (i < readings.size() - 1) log_file << ", ";
            }
            log_file << "]\n";
        }
        log_file.close();
    }

    // Publish driving commands to steer towards the best point
    void publish_drive_message(float angle, float speed) {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.drive.steering_angle = angle;
        msg.drive.speed = speed;
        publisher_->publish(msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
    float max_angle_, angle_min_, angle_increment_, scan_ranges_max_, car_width_;
};

int main(int argc, char *argv[]) {
    // Initialize log file by clearing previous content
    std::ofstream log_file(readings_log_file);
    log_file.close();

    // Initialize ROS 2 and create the reactive node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReactiveFollowGap>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
