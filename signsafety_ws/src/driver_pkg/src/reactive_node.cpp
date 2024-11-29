#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>

class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node") {
        // Declare and retrieve parameters
        this->declare_parameter("fov_range", 75.0);
        this->declare_parameter("longest_range_threshold", 7.0);
        this->declare_parameter("car_width", 0.5);
        this->declare_parameter("min_speed", 0.1);
        this->declare_parameter("max_speed", 2.0);
        this->declare_parameter("stop_distance_threshold", 1.5);
        this->declare_parameter("yield_speed", 0.5);

        fov_range_ = this->get_parameter("fov_range").as_double();
        longest_range_threshold_ = this->get_parameter("longest_range_threshold").as_double();
        car_width_ = this->get_parameter("car_width").as_double();
        min_speed_ = this->get_parameter("min_speed").as_double();
        max_speed_ = this->get_parameter("max_speed").as_double();
        stop_distance_threshold_ = this->get_parameter("stop_distance_threshold").as_double();
        yield_speed_ = this->get_parameter("yield_speed").as_double();

        // Subscribe to topics and initialize publishers
        lidar_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1));

        sign_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/sign_info", 10, std::bind(&ReactiveFollowGap::sign_callback, this, std::placeholders::_1));

        drive_publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);
    }

private:
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr data) {
        if (processing_sign_) {
            return;  // Skip gap-following logic while processing a sign
        }

        std::vector<float> ranges = data->ranges;
        auto processed_ranges = preprocess_lidar(ranges);

        auto [start_i, end_i] = find_max_gap(processed_ranges);
        if (start_i == -1) {
            RCLCPP_WARN(this->get_logger(), "No valid gaps found. Stopping the vehicle.");
            publish_drive_message(0.0, 0.0);
            return;
        }

        int best_point = find_best_point(start_i, end_i, processed_ranges);
        float angle = data->angle_min + best_point * data->angle_increment;
        float speed = calculate_speed(processed_ranges[best_point]);

        publish_drive_message(angle, speed);
    }

    void sign_callback(const std_msgs::msg::String::SharedPtr msg) {
        std::istringstream ss(msg->data);
        std::string sign_type;
        float distance;

        if (std::getline(ss, sign_type, ',') && ss >> distance) {
            RCLCPP_INFO(this->get_logger(), "Sign detected: %s, Distance: %.2f", sign_type.c_str(), distance);
            handle_sign(sign_type, distance);
        }
    }

    void handle_sign(const std::string& sign_type, float distance) {
        if (sign_type == "STOP" && distance <= stop_distance_threshold_) {
            RCLCPP_WARN(this->get_logger(), "STOP sign detected. Halting vehicle.");
            publish_drive_message(0.0, 0.0);
            processing_sign_ = true;

        } else if (sign_type == "SPEED_LIMIT") {
            float speed_limit = std::clamp(distance, min_speed_, max_speed_);
            RCLCPP_INFO(this->get_logger(), "Adjusting speed to %.2f due to SPEED_LIMIT.", speed_limit);
            publish_drive_message(0.0, speed_limit);
            processing_sign_ = true;

        } else if (sign_type == "YIELD") {
            RCLCPP_INFO(this->get_logger(), "Yield sign detected. Reducing speed.");
            publish_drive_message(0.0, yield_speed_);
            processing_sign_ = true;
            
        } else {
            processing_sign_ = false;  // Resume normal operation for unrecognized signs
        }
    }

    std::vector<float> preprocess_lidar(const std::vector<float>& ranges) {
        std::vector<float> filtered_ranges = ranges;
        std::replace_if(filtered_ranges.begin(), filtered_ranges.end(),
                        [](float range) { return std::isnan(range) || std::isinf(range); }, 0.0);
        return filtered_ranges;
    }

    std::pair<int, int> find_max_gap(const std::vector<float>& ranges) {
        int start = -1, end = -1;
        int max_start = -1, max_end = -1, max_size = -1;
        for (int i = 0; i < ranges.size(); ++i) {
            if (ranges[i] > car_width_) {
                if (start == -1) {
                    start = i;
                }
                end = i;
            } else {
                if (start != -1 && end - start > max_size) {
                    max_size = end - start;
                    max_start = start;
                    max_end = end;
                }
                start = -1;
                end = -1;
            }
        }
        if (start != -1 && end - start > max_size) {
            max_start = start;
            max_end = end;
        }
        return {max_start, max_end};
    }

    int find_best_point(int start_i, int end_i, const std::vector<float>& ranges) {
        float max_value = -std::numeric_limits<float>::infinity();
        int best_point = start_i;
        for (int i = start_i; i <= end_i; ++i) {
            if (ranges[i] > max_value) {
                max_value = ranges[i];
                best_point = i;
            }
        }
        return best_point;
    }

    float calculate_speed(float distance) {
        return std::clamp(min_speed_ + (distance / longest_range_threshold_) * (max_speed_ - min_speed_), min_speed_, max_speed_);
    }

    void publish_drive_message(float angle, float speed) {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.drive.steering_angle = angle;
        msg.drive.speed = speed;
        drive_publisher_->publish(msg);
    }

    // ROS 2 communication
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sign_subscription_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher_;

    // Parameters
    float fov_range_, longest_range_threshold_, car_width_;
    float min_speed_, max_speed_;
    float stop_distance_threshold_, yield_speed_;
    bool processing_sign_ = false;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReactiveFollowGap>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
