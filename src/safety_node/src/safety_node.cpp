#include "rclcpp/rclcpp.hpp"
/// CHECK: include needed ROS msg type headers and libraries
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"


class Safety : public rclcpp::Node {
// The class that handles emergency braking

public:
    Safety() : 
    Node("safety_node"),
    odom_sub_(this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&Safety::drive_callback, this, std::placeholders::_1))),
    scan_sub_(this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&Safety::scan_callback, this, std::placeholders::_1))),
    drive_pub_(this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10))
    {
        /*
        You should also subscribe to the /scan topic to get the
        sensor_msgs/LaserScan messages and the /ego_racecar/odom topic to get
        the nav_msgs/Odometry messages

        The subscribers should use the provided odom_callback and 
        scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        */

        /// TODO: create ROS subscribers and publishers
        
    }

private:
    double speed = 0.0;      // longitudinal velocity v_x (odom linear.x)
    double yaw_rate = 0.0;   // yaw rate psi_dot (odom angular.z)

    static constexpr double lr_ = 0.275;

    // Require this many consecutive over-threshold scans before braking (debounce).
    static constexpr int brake_trigger_count_ = 5;
    int brake_counter_ = 0;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    /// TODO: create ROS subscribers and publishers

    void drive_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        /// TODO: update current speed
        speed = msg->twist.twist.linear.x;
        yaw_rate = msg->twist.twist.angular.z;

    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {
        /// TODO: calculate TTC

        const auto ittc = calculate_ttc(scan_msg);
        const auto ttc_threshold = 2.0;
        
        if(ittc < ttc_threshold) {
            brake_counter_++;
        } else {
            brake_counter_ = 0;
        }

        if(brake_counter_ >= brake_trigger_count_) {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);

            setenv("TZ", "America/New_York", 1);
            tzset();

            std::stringstream ss;
            ss << std::put_time(std::localtime(&t), "%H:%M:%S");

            RCLCPP_WARN(
                this->get_logger(),
                "Emergency braking! Time: %s, TTC: %.3f s with threshold %.3f",
                ss.str().c_str(),
                ittc,
                ttc_threshold
                
            );            
            // Publish a drive message with zero speed to stop the car
            auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
            drive_msg.drive.speed = 0.0;
            drive_pub_->publish(drive_msg);
        }



        /// TODO: publish drive/brake message
    }

    double calculate_ttc(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        auto min_ttc = std::numeric_limits<double>::infinity();

        const auto v_x = speed;
        const auto v_y = lr_ * yaw_rate;

        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            const auto range = scan_msg->ranges[i];
            if (std::isnan(range) || std::isinf(range) ||
                range < scan_msg->range_min || range > scan_msg->range_max) {
                continue; 
            }

            const auto angle = scan_msg->angle_min + i * scan_msg->angle_increment;

            const auto range_rate = v_x * std::cos(angle) + v_y * std::sin(angle);

            if (range_rate > 0) { 
                double ttc = range / range_rate;
                if (ttc < min_ttc) {
                    min_ttc = ttc;
                }
            }
        }
        return min_ttc;
    }



};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Safety>());
    rclcpp::shutdown();
    return 0;
}