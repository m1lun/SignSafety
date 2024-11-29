#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>

// ReactiveFollowGap Node
class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node") {
        // Declare ROS 2 parameters
        this->declare_parameter("fov_range", 75.0);
        this->declare_parameter("longest_range_threshold", 7.0);
        this->declare_parameter("car_width", 0.5);
        this->declare_parameter("min_speed", 0.1);
        this->declare_parameter("max_speed", 2.0);
        this->declare_parameter("log_file_path", "/tmp/lidar_readings.log");

        // Fetch parameters
        fov_range_ = this->get_parameter("fov_range").as_double();
        longest_range_threshold_ = this->get_parameter("longest_range_threshold").as_double();
        car_width_ = this->get_parameter("car_width").as_double();
        min_speed_ = this->get_parameter("min_speed").as_double();
        max_speed_ = this->get_parameter("max_speed").as_double();
        log_file_path_ = this->get_parameter("log_file_path").as_string();

        // Topics
        auto lidarscan_topic = "/scan";
        auto drive_topic = "/drive";

        // LIDAR subscription
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidarscan_topic, 10, std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1));

        // Drive publisher
        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);
    }

private:
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr data) {
        std::vector<float> ranges = data->ranges;
        angle_min_ = data->angle_min;
        angle_increment_ = data->angle_increment;

        // Preprocess LIDAR readings
        auto processed_ranges = preprocess_lidar(ranges);

        // Check for gaps
        auto [start_i, end_i] = find_max_gap(processed_ranges);
        if (start_i == -1) {
            RCLCPP_WARN(this->get_logger(), "No valid gaps found. Stopping the vehicle.");
            publish_drive_message(0.0, 0.0);
            return;
        }

        // Best point selection
        int best_point = find_best_point(start_i, end_i, processed_ranges);

        // Calculate angle and speed
        float angle = angle_min_ + best_point * angle_increment_;
        float speed = calculate_speed(processed_ranges[best_point]);

        // Log driving details
        RCLCPP_INFO(this->get_logger(), "Steering angle: %.2f, Speed: %.2f", angle, speed);

        // Publish drive message
        publish_drive_message(angle, speed);
    }

    std::vector<float> preprocess_lidar(const std::vector<float>& ranges) {
        float fov_limit = fov_range_ * M_PI / 180.0;
        int start_index = std::max(0, static_cast<int>((-fov_limit - angle_min_) / angle_increment_));
        int end_index = std::min(static_cast<int>(ranges.size()), static_cast<int>((fov_limit - angle_min_) / angle_increment_));

        std::vector<float> processed(ranges.size(), 0.0);
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i] > longest_range_threshold_) ranges[i] = longest_range_threshold_;
            if (std::isnan(ranges[i]) || i < start_index || i > end_index) {
                processed[i] = 0.0;
            } else {
                processed[i] = ranges[i];
            }
        }
        log_readings(processed);
        return processed;
    }

    std::pair<int, int> find_max_gap(const std::vector<float>& ranges) {
        int max_gap = 0, start_i = -1, end_i = -1;
        int i = 0;

        while (i < ranges.size()) {
            while (i < ranges.size() && ranges[i] == 0.0) i++;
            int j = i;
            while (j < ranges.size() && ranges[j] > 0.0) j++;
            int gap_size = j - i;
            if (gap_size > max_gap) {
                max_gap = gap_size;
                start_i = i;
                end_i = j - 1;
            }
            i = j;
        }
        return {start_i, end_i};
    }

    int find_best_point(int start_i, int end_i, const std::vector<float>& ranges) {
        auto max_it = std::max_element(ranges.begin() + start_i, ranges.begin() + end_i + 1);
        return std::distance(ranges.begin(), max_it);
    }

    float calculate_speed(float distance) {
        return std::clamp(min_speed_ + (distance / longest_range_threshold_) * (max_speed_ - min_speed_), min_speed_, max_speed_);
    }

    void log_readings(const std::vector<float>& readings) {
        std::ofstream log_file(log_file_path_, std::ios::app);
        if (log_file.is_open()) {
            log_file << "[" << std::fixed;
            for (size_t i = 0; i < readings.size(); ++i) {
                log_file << readings[i] << (i < readings.size() - 1 ? ", " : "");
            }
            log_file << "]\n";
            log_file.close();
        }
    }

    void publish_drive_message(float angle, float speed) {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.drive.steering_angle = angle;
        msg.drive.speed = speed;
        publisher_->publish(msg);
    }

    // Node variables
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
    float fov_range_, longest_range_threshold_, car_width_, min_speed_, max_speed_;
    float angle_min_, angle_increment_;
    std::string log_file_path_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReactiveFollowGap>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
