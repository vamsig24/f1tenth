#include <sstream>
#include <string>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <tf2/utils.h>
#include <fstream>

#include "pure_pursuit/waypoint_utils.hpp"


/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

class PurePursuit : public rclcpp::Node
{
    // Implement PurePursuit
    // This is just a template, you are free to implement your own node!
public:
    PurePursuit() : Node("pure_pursuit_node"),
    pose_sub_(create_subscription<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 10, std::bind(&PurePursuit::OnPose, this, std::placeholders::_1))),
    path_sub_(create_subscription<nav_msgs::msg::Path>("/rrt_path", 10, std::bind(&PurePursuit::OnPath, this, std::placeholders::_1))),
    drive_pub_(create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10)),
    waypoint_pub_(create_publisher<visualization_msgs::msg::MarkerArray>("/waypoints", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local())),
    lookahead_pub_(create_publisher<visualization_msgs::msg::Marker>("/lookahead_point", 10))
    {
        waypoints_ = pure_pursuit::LoadCSV("/sim_ws/src/lap_timer/waypoints/levine.csv");
        waypoints_in_execution_ = waypoints_;

        std::string odom_topic = "/odom";
        
        if(use_sim_){
         odom_topic = "ego_racecar/odom";
        }

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, 10, std::bind(&PurePursuit::OnOdom, this, std::placeholders::_1));

        std::cout << "Loaded waypoints " << waypoints_.size() << " from CSV" << std::endl;

        // RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());

        PublishWaypoints();
    
    }
    ~PurePursuit() {}


private:

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr lookahead_pub_;
    std::vector<geometry_msgs::msg::Point> waypoints_;
    std::vector<geometry_msgs::msg::Point> waypoints_in_execution_;
    float lookahead_distance_{0.5F};
    bool use_sim_{true};
    nav_msgs::msg::Odometry current_odom_;
    static constexpr double WHEELBASE {0.3302};
    static constexpr double MAX_STEER {0.34};
    static constexpr double MIN_SPEED {1.};
    static constexpr double MAX_SPEED {9.5};
    static constexpr double MIN_LOOKAHEAD_DISTANCE {0.35F};
    static constexpr double MAX_LOOKAHEAD_DISTANCE {1.3F};
    static constexpr double PATH_RESOLUTION {0.1};



private:
    // Resamples a polyline so consecutive points are spaced ~resolution metres apart.
    static std::vector<geometry_msgs::msg::Point> ResamplePath(const std::vector<geometry_msgs::msg::Point>& pts, double resolution)
    {
        std::vector<geometry_msgs::msg::Point> out;
        if (pts.size() < 2) {
            return pts;
        }

        out.push_back(pts.front());
        for (size_t i = 1; i < pts.size(); ++i) {
            const double dx = pts[i].x - pts[i - 1].x;
            const double dy = pts[i].y - pts[i - 1].y;
            const double seg = std::hypot(dx, dy);
            if (seg < 1e-9) {
                continue;
            }
            const double ux = dx / seg;
            const double uy = dy / seg;
            for (double d = resolution; d <= seg; d += resolution) {
                geometry_msgs::msg::Point p;
                p.x = pts[i - 1].x + ux * d;
                p.y = pts[i - 1].y + uy * d;
                out.push_back(p);
            }
        }
        out.push_back(pts.back());

        return out;
    }

    static double InterpolateSpeed(double steer, double min_speed, double max_speed, double max_steer)
    {
        double ratio = std::clamp(std::abs(steer) / max_steer, 0.0, 1.0);

        return max_speed + ratio * (min_speed - max_speed);
    }

    static double InterpolateLookahead(double speed, double min_lookahead, double max_lookahead, double max_speed)
    {
        double ratio = std::clamp(speed / max_speed, 0.0, 1.0);

        return min_lookahead + ratio * (max_lookahead - min_lookahead);
    }


    void ExecutePurepursuit(const geometry_msgs::msg::Pose &current_pose) 
    {
        const auto &q = current_pose.orientation;
        const double quat_norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if (quat_norm_sq < 1e-6) {
            RCLCPP_WARN(this->get_logger(),
                "Skipping: invalid orientation quaternion (%.3f %.3f %.3f %.3f)", q.x, q.y, q.z, q.w);
            return;
        }

        // Track the reference that has the latest RRT detour spliced into it.
        const auto &waypoints = waypoints_in_execution_;

        // TODO: find the current waypoint to track using methods mentioned in lecture
        const auto current_pose_waypoint_idx = pure_pursuit::GetIndexClosestToPoint(current_pose.position, waypoints);

        lookahead_distance_ = InterpolateLookahead(current_odom_.twist.twist.linear.x, MIN_LOOKAHEAD_DISTANCE, MAX_LOOKAHEAD_DISTANCE, MAX_SPEED);

        const auto goal_waypoint_idx_opt = pure_pursuit::GetClosestIndexAtDistance(lookahead_distance_, current_pose_waypoint_idx, waypoints);

        if(!goal_waypoint_idx_opt.has_value()) {
            RCLCPP_WARN(this->get_logger(), "No valid goal waypoint found at the specified lookahead distance.");

            ackermann_msgs::msg::AckermannDriveStamped drive_msg;
            drive_msg.header.stamp = now();
            drive_msg.header.frame_id = "base_link";
            drive_msg.drive.steering_angle = 0;
            drive_msg.drive.speed = 0;
            drive_pub_->publish(drive_msg);

            return;
        }

        const auto goal_waypoint_idx = goal_waypoint_idx_opt.value();

        // TODO: transform goal point to vehicle frame of reference
        const auto &goal_point = waypoints[goal_waypoint_idx];
        const auto actual_lookahead_distance = std::hypot(goal_point.x - current_pose.position.x, goal_point.y - current_pose.position.y);

        PublishLookahead(goal_point);
    
        double yaw = tf2::getYaw(current_pose.orientation);

        const double dx = goal_point.x - current_pose.position.x;
        const double dy = goal_point.y - current_pose.position.y;
        const double goal_x_vehicle = std::cos(yaw) * dx + std::sin(yaw) * dy;
        const double goal_y_vehicle = -std::sin(yaw) * dx + std::cos(yaw) * dy;

        const auto curvature = 2 * goal_y_vehicle / (actual_lookahead_distance * actual_lookahead_distance);

        const auto steering_angle = std::clamp(std::atan(curvature * WHEELBASE), -MAX_STEER, MAX_STEER);

        // RCLCPP_INFO(this->get_logger(),
        //     "pose=(%.3f, %.3f) yaw=%.3f | closest_idx=%zu goal_idx=%zu goal=(%.3f, %.3f) | "
        //     "L=%.3f gx=%.3f gy=%.3f curv=%.3f steer=%.3f",
        //     current_pose.position.x, current_pose.position.y, yaw,
        //     current_pose_waypoint_idx, goal_waypoint_idx, goal_point.x, goal_point.y,
        //     actual_lookahead_distance, goal_x_vehicle, goal_y_vehicle, curvature, steering_angle);

        // TODO: publish drive message, don't forget to limit the steering angle.
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = steering_angle;
        drive_msg.drive.speed = InterpolateSpeed(steering_angle, MIN_SPEED, MAX_SPEED, MAX_STEER);

        // RCLCPP_INFO(this->get_logger(), "publishing steer=%.3f speed=%.3f",
        //     drive_msg.drive.steering_angle, drive_msg.drive.speed);

        drive_pub_->publish(drive_msg);


    }



    void PublishLookahead(const geometry_msgs::msg::Point& point) const
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = now();
        marker.ns = "lookahead";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = point;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        lookahead_pub_->publish(marker);
    }

    void PublishWaypoints()
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = now();
        marker.ns = "waypoints";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::POINTS;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.color.a = 1.0;
        marker.color.g = 1.0;
        marker.points = waypoints_;

        marker_array.markers.push_back(marker);
        waypoint_pub_->publish(marker_array);
    }

    void OnOdom(const nav_msgs::msg::Odometry::ConstPtr &odom_msg)
    {
        // RCLCPP_INFO(this->get_logger(), "OnOdom fired");
        // TODO: handle odometry updates
        if (use_sim_) {
            ExecutePurepursuit(odom_msg->pose.pose);
        }

        current_odom_ = *odom_msg;
    }

    void OnPose(const geometry_msgs::msg::PoseStamped::ConstPtr &pose_msg)
    {
        // RCLCPP_INFO(this->get_logger(), "OnPose fired");
        ExecutePurepursuit(pose_msg->pose);
        
    }

    void OnPath(const nav_msgs::msg::Path::ConstPtr &path_msg)
    {
        if (path_msg->poses.empty()) {
            return;
        }

        std::vector<geometry_msgs::msg::Point> pts;
        pts.reserve(path_msg->poses.size());
        for (const auto &pose : path_msg->poses) {
            pts.push_back(pose.pose.position);
        }

        const auto rrt_path = ResamplePath(pts, PATH_RESOLUTION);

        auto start_idx = pure_pursuit::GetIndexClosestToPoint(rrt_path.front(), waypoints_);
        auto end_idx = pure_pursuit::GetIndexClosestToPoint(rrt_path.back(), waypoints_);
        if (start_idx > end_idx) {
            std::swap(start_idx, end_idx);
        }

        // Rebuild from the untouched original, replacing the spanned segment with the RRT detour.
        auto reference = waypoints_;
        const auto splice_begin = reference.begin() + static_cast<std::ptrdiff_t>(start_idx);
        const auto splice_end = reference.begin() + static_cast<std::ptrdiff_t>(end_idx) + 1;
        const auto insert_at = reference.erase(splice_begin, splice_end);
        reference.insert(insert_at, rrt_path.begin(), rrt_path.end());

        waypoints_in_execution_ = std::move(reference);
    }

};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}