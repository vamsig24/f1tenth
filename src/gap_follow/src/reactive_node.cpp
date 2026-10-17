#include "rclcpp/rclcpp.hpp"
#include <vector>
#include <deque>
#include <utility>
#include <cmath>
#include <algorithm>
#include <string>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"

class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node"),
        scan_sub_(create_subscription<sensor_msgs::msg::LaserScan>("/scan", 10, std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1))),
        odom_sub_(create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 10, std::bind(&ReactiveFollowGap::odom_callback, this, std::placeholders::_1))),
        drive_pub_(create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10)),
        bubble_pub_(create_publisher<visualization_msgs::msg::Marker>("/gap_follow/bubble", 10)),
        goal_pub_(create_publisher<visualization_msgs::msg::Marker>("/gap_follow/goal_point", 10)),
        fixed_scan_pub_(create_publisher<sensor_msgs::msg::LaserScan>("/gap_follow/fixed_scan", 10))
    {}

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr bubble_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr fixed_scan_pub_;

    static constexpr float kMaxRange = 5.0f;  
    static constexpr int kWindow = 5;         
    static constexpr float kBubble = 0.75f;    
    static constexpr float kFov = 1.57f;       

    static constexpr float kDisparity = 0.25f;     
    static constexpr float kCarHalfWidth = 0.15f; 
    static constexpr float kTolerance = 0.1f;   

    static constexpr float kSideThreshold = 0.2f; 

    static constexpr double kD0 = 1.0, kV0 = 0.5;
    static constexpr double kD1 = 2.5, kV1 = 1.0;
    static constexpr double kD2 = 5.0, kV2 = 2.0;

    static constexpr size_t kMaxWindow = 6;   
    std::deque<std::pair<float, float>> goal_history_;   
    float current_speed_ = 0.0f;  

    void lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
    {

        std::vector<float> ranges(scan->ranges.begin(), scan->ranges.begin() + scan->ranges.size() - 1 );

        fix_disparities(ranges, scan->angle_increment);

        publish_fixed_scan(*scan, ranges);

        preprocess_lidar(ranges);

        const int closest = std::min_element(ranges.begin(), ranges.end()) - ranges.begin();
        const auto bubble = static_cast<int>(std::atan2(kBubble, ranges[closest]) / scan->angle_increment);
        const auto b_start = std::max(0, closest - bubble);
        const auto b_end = std::min<int>(ranges.size() - 1, closest + bubble);
        std::vector<geometry_msgs::msg::Point> bubble_pts;

        for (int i = b_start; i <= b_end; ++i) {
            float ang = scan->angle_min + i * scan->angle_increment;
            bubble_pts.push_back(to_point(ranges[i], ang));
            ranges[i] = 0.0f;
        }

        const auto [gap_start, gap_end] = find_max_gap(ranges);

        const auto best_local = find_best_point(ranges, gap_start, gap_end);
        const auto best_angle = scan->angle_min + best_local * scan->angle_increment;

        publish_bubble_marker(bubble_pts, scan->header.frame_id);

        const auto force_straight = side_too_close(*scan);

        publish_drive(best_angle, ranges[best_local], scan->header.frame_id, force_straight);
    }

    bool side_too_close(const sensor_msgs::msg::LaserScan& scan) const
    {
        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            float angle = scan.angle_min + i * scan.angle_increment;

            if (angle < -static_cast<float>(M_PI) || angle >= static_cast<float>(M_PI)) 
            {
                continue;
            }   

            float r = scan.ranges[i];

            if (r > 0.0f && r < kSideThreshold) 
            {
                return true;
            }
        }
        return false;
    }

    void fix_disparities(std::vector<float>& ranges, float angle_increment)
    {

        const auto n = static_cast<int>(ranges.size());
        std::vector<float> out = ranges;

        for (int i = 0; i < n - 1; ++i) {
            float a = ranges[i];
            float b = ranges[i + 1];

            if (std::abs(a - b) < kDisparity) continue;

            const auto closer = std::min(a, b);

            if (closer <= 0.0f) continue;
            
            const auto extend = static_cast<int>(std::ceil((kCarHalfWidth + kTolerance) / (closer * angle_increment)));

            if (a < b) {
                for (int j = i + 1; j <= std::min(n - 1, i + extend); ++j)
                    out[j] = std::min(out[j], closer);
            } else {
                for (int j = i; j >= std::max(0, i + 1 - extend); --j)
                    out[j] = std::min(out[j], closer);
            }
        }

        ranges = std::move(out);
    }

    static geometry_msgs::msg::Point to_point(float range, float angle)
    {
        geometry_msgs::msg::Point p;
        p.x = range * std::cos(angle);
        p.y = range * std::sin(angle);
        p.z = 0.0;

        return p;
    }

    void publish_bubble_marker(const std::vector<geometry_msgs::msg::Point>& pts, const std::string& frame)
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame;
        m.header.stamp = now();
        m.ns = "bubble";
        m.type = visualization_msgs::msg::Marker::POINTS;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.08;
        m.scale.y = 0.08;
        m.color.r = 1.0f;
        m.color.a = 1.0f;
        m.points = pts;
        bubble_pub_->publish(m);
    }

    void publish_goal_marker(const geometry_msgs::msg::Point& pt, const std::string& frame)
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame;
        m.header.stamp = now();
        m.ns = "goal";
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position = pt;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.25;
        m.color.g = 1.0f;
        m.color.a = 1.0f;
        goal_pub_->publish(m);
    }

    void publish_fixed_scan(const sensor_msgs::msg::LaserScan& scan, const std::vector<float>& ranges)
    {
        sensor_msgs::msg::LaserScan out = scan;   
        out.ranges.assign(ranges.begin(), ranges.end());
        out.intensities.clear();
        fixed_scan_pub_->publish(out);
    }

    static int angle_to_index(const sensor_msgs::msg::LaserScan& scan, float angle)
    {
        return static_cast<int>((angle - scan.angle_min) / scan.angle_increment);
    }

     void preprocess_lidar(std::vector<float>& ranges)
    {
        std::vector<float> out(ranges.size());

        for (int i = 0; i < static_cast<int>(ranges.size()); ++i) {
            float sum = 0.0f;
            int count = 0;
            for (int j = std::max(0, i - kWindow); j <= std::min<int>(ranges.size() - 1, i + kWindow); ++j) {
                float v = ranges[j];

                if(std::isnan(v)){
                    sum += 0.0f;
                } else if(std::isinf(v)){
                    sum+= kMaxRange;
                }else {
                    sum += v;
                }

                ++count;
            }
            out[i] = std::min(sum / count, kMaxRange);
        }

        ranges = std::move(out);
    }

    static std::pair<int, int> find_max_gap(const std::vector<float>& ranges)
    {
        int best_start = 0, best_len = 0, cur_start = 0, cur_len = 0;

        for (int i = 0; i < static_cast<int>(ranges.size()); ++i) {
            if (ranges[i] > 0.0f) {
                if (cur_len++ == 0) cur_start = i;
                if (cur_len > best_len) { best_len = cur_len; best_start = cur_start; }
            } else {
                cur_len = 0;
            }
        }

        return {best_start, best_start + best_len - 1};
    }

    static int find_best_point(const std::vector<float>& ranges, int start, int end)
    {
        int best = start;

        for (int i = start; i <= end; ++i) {
            if (ranges[i] > ranges[best]) best = i;
        }

        return best;
    }

    double GetRadFromDeg(double deg)
    {
        return deg * M_PI / 180.0;
    }

    double GetSpeedFromGoalDist(double dist)
    {
        if (dist <= kD0) return kV0 ;
        if (dist <= kD1) return kV0 + (kV1 - kV0) * (dist - kD0) / (kD1 - kD0);
        if (dist <= kD2) return kV1 + (kV2 - kV1) * (dist - kD1) / (kD2 - kD1);

        return kV2;
    }

    size_t window_from_speed(double speed) const
    {
        if (speed < 1.0) return 1;
        if (speed < 1.5) return 3;

        return 6;
    }

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        current_speed_ = msg->twist.twist.linear.x;
    }

    void publish_drive(float angle, float goal_dist, const std::string& frame, bool force_straight)
    {
        goal_history_.push_back({angle, goal_dist});

        while (goal_history_.size() > kMaxWindow) goal_history_.pop_front();

        size_t window = std::min(window_from_speed(current_speed_), goal_history_.size());
        float sum_angle = 0.0f, sum_dist = 0.0f;

        for (auto it = goal_history_.end() - window; it != goal_history_.end(); ++it) {
            sum_angle += it->first;
            sum_dist += it->second;
        }

        float smoothed_angle = sum_angle / window;
        float smoothed_dist = sum_dist / window;

        if (force_straight) smoothed_angle = 0.0f;

        publish_goal_marker(to_point(smoothed_dist, smoothed_angle), frame);

        float speed = GetSpeedFromGoalDist(smoothed_dist);

        ackermann_msgs::msg::AckermannDriveStamped msg;
        msg.header.stamp = now();
        msg.header.frame_id = "base_link";
        msg.drive.steering_angle = smoothed_angle;
        msg.drive.speed = speed;
        drive_pub_->publish(msg);
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}