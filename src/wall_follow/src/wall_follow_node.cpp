#include "rclcpp/rclcpp.hpp"
#include <string>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <cmath>

class WallFollow : public rclcpp::Node {

public:
    WallFollow() : Node("wall_follow_node"),
    scan_sub_(this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&WallFollow::scan_callback, this, std::placeholders::_1))),
    drive_pub_(this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10))
  
    {
        prev_timestamp_ = std::chrono::steady_clock::now();
        // TODO: create ROS subscribers and publishers
    }

private:
    // PID CONTROL PARAMS
    // TODO: double kp =
    // TODO: double kd =
    // TODO: double ki =
    double kp_ = 1.5;
    double kd_ = 2.0;
    double ki_ = 0.0;
    double servo_offset_ = 0.0;
    double prev_error_ = 0.0;
    double error_ = 0.0;
    double integral_ = 0.0;
    double angle_min_ = 0.0;
    double angle_max_ = 0.0;
    double angle_increment_ = 0.0;
    double range_count_ = 0.0;
    const double lookahead_dist_ = 1.0; // meters
    const double desired_distance_ = 0.75; // meters
    std::chrono::steady_clock::time_point prev_timestamp_;
  
    // Topics
    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";
    /// TODO: create ROS subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;


     double GetRadFromDeg(double deg)
    {
        /*
        Simple helper to convert degrees to radians

        Args:
            deg: angle in degrees

        Returns:
            rad: angle in radians
        */
        return deg * M_PI / 180.0;
    }

    double GetSpeedFromCmdSteer(double cmd_steer)
    {
        /*
        Simple helper to return the corresponding speed based on the steering angle. You can use this to slow down when turning.

        Args:
            cmd_steer: steering angle command

        Returns:
            speed: speed command
        */

        const auto abs_cmd_steer = std::abs(cmd_steer);

        const double steer_threshold_1 = GetRadFromDeg(10.0); // 10 degrees in radians
        const double steer_threshold_2 = GetRadFromDeg(20.0); // 20 degrees in radians

       
        double speed = 0.5; // default speed

        if (abs_cmd_steer > 0 && abs_cmd_steer < steer_threshold_1  ) {
            speed = 1.5; // slow down when turning sharply
        }else if (abs_cmd_steer >= steer_threshold_1 && abs_cmd_steer < steer_threshold_2) {
            speed = 1.0; // slow down when turning sharply
        }

        return speed;
    }

    double get_range(const float* range_data, double angle)
    {
        /*
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle
        */

        if( angle < angle_min_ || angle > angle_max_) {
            RCLCPP_ERROR(this->get_logger(), "Angle out of bounds");
            return 0.0;
        }

        size_t index = static_cast<size_t>((angle - angle_min_) / angle_increment_);
        double range = range_data[index];

        if (std::isnan(range) || std::isinf(range)) {
            RCLCPP_ERROR(this->get_logger(), "Invalid range for input angle: %f, so returning 0.0", angle);
            return 0.0;
        }

        return range;
    }

    double get_error(const float* range_data, double dist)
    {
        /*
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        */

        static constexpr double b_angle = 90.0; // degrees
        static constexpr double a_angle = 45.0; // degrees

        static constexpr double b_radians = b_angle * M_PI / 180.0;
        static constexpr double a_radians = a_angle * M_PI / 180.0;

        const auto b = get_range(range_data, b_radians);
        const auto a = get_range(range_data, a_radians);

        const auto vehicle_heading = std::atan2(a * std::cos(a_radians) - b, a * std::sin(a_radians));
        const auto distance_to_wall = b * std::cos(vehicle_heading);
        const auto projected_distance = distance_to_wall + lookahead_dist_ * std::sin(vehicle_heading); // 
        // TODO:implement
        return projected_distance - dist;
    }

    void pid_control(double error, double velocity)
    {
        /*
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        */

        const auto current_timestamp = std::chrono::steady_clock::now();
        const auto dt = std::chrono::duration<double>(current_timestamp - prev_timestamp_).count();

        prev_timestamp_ = current_timestamp;
        integral_ += error * (dt + 1e-6);
        const auto differential_error = (error - prev_error_) / (dt + 1e-6); 
        const auto cmd_steer = kp_ * error + kd_ * differential_error + ki_ * integral_;
        const auto corrected_cmd_steer = cmd_steer + servo_offset_;
        prev_error_ = error;
        // TODO: Use kp, ki & kd to implement a PID controller
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.steering_angle = corrected_cmd_steer;
        drive_msg.drive.speed = GetSpeedFromCmdSteer(drive_msg.drive.steering_angle);
        // TODO: fill in drive message and publish
        drive_pub_->publish(drive_msg);
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {
        angle_max_ = scan_msg->angle_max;
        angle_min_ = scan_msg->angle_min;
        range_count_ = scan_msg->ranges.size();
        angle_increment_ = scan_msg->angle_increment;
        /*
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        */
        double error = get_error(scan_msg->ranges.data(), desired_distance_); // TODO: replace with error calculated by get_error()
        double velocity = 0.0; // TODO: calculate desired car velocity based on error
        // TODO: actuate the car with PID
        pid_control(error, velocity);

    }

};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallFollow>());
    rclcpp::shutdown();
    return 0;
}