#include <sstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_msgs/msg/float32.hpp"
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


using namespace std;

// Pure pursuit with no hand-tuned speed -> lookahead table.
//
// Speed: the fastest speed from which the car can still brake down to every
//   curvature limit ahead, sqrt(a_lat / kappa), within max_decel. max_speed is
//   only a cap, and RRT's free distance (how far ahead the path is clear) caps
//   it again so the car never outruns what the planner has cleared.
// Lookahead: as long as the path stays within chord_tolerance of the straight
//   line to the lookahead point, bounded to [lookahead_time_min, lookahead_time_max]
//   seconds of travel and to [lookahead_min, lookahead_max] metres. Long on
//   straights, short through a detour, and it rescales on its own when
//   max_speed changes.
class PurePursuit : public rclcpp::Node
{
public:
    PurePursuit() : Node("pure_pursuit_node"),
    pose_sub_(create_subscription<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 10, std::bind(&PurePursuit::OnPose, this, std::placeholders::_1))),
    path_sub_(create_subscription<nav_msgs::msg::Path>("/rrt_path", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(), std::bind(&PurePursuit::OnPath, this, std::placeholders::_1))),
    free_distance_sub_(create_subscription<std_msgs::msg::Float32>("/rrt/free_distance", 10, std::bind(&PurePursuit::OnFreeDistance, this, std::placeholders::_1))),
    drive_pub_(create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10)),
    waypoint_pub_(create_publisher<visualization_msgs::msg::MarkerArray>("/waypoints", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local())),
    lookahead_pub_(create_publisher<visualization_msgs::msg::Marker>("/lookahead_point", 10))
    {
        use_sim_ = declare_parameter("use_sim", false);
        // 9.5 m/s is a sim number; the car's lookahead below is tuned at 2.5
        max_speed_ = declare_parameter("max_speed", use_sim_ ? 9.5 : 2.5);
        min_speed_ = declare_parameter("min_speed", 1.0);
        max_lat_accel_ = declare_parameter("max_lat_accel", 8.0);
        // braking measured on the car's bags is 3-5 m/s^2 (median ~4.2); 8 is a sim number
        max_decel_ = declare_parameter("max_decel", use_sim_ ? 8.0 : 4.0);
        lookahead_min_ = declare_parameter("lookahead_min", 0.5);
        lookahead_max_ = declare_parameter("lookahead_max", std::numeric_limits<double>::infinity());
        // The car's steering responds ~170 ms after the command, and pure pursuit
        // weaves when that is a large share of the lookahead. 0.6-1.0 s held the car
        // within ~6 cm (max 21 cm) of the raceline at 2.5 m/s; 0.2-0.45 s weaved +-0.5 m.
        lookahead_time_min_ = declare_parameter("lookahead_time_min", use_sim_ ? 0.2 : 0.6);
        lookahead_time_max_ = declare_parameter("lookahead_time_max", use_sim_ ? 0.45 : 1.0);
        chord_tolerance_ = declare_parameter("chord_tolerance", 0.08);
        stop_margin_ = declare_parameter("stop_margin", 0.5);
        free_distance_timeout_ = declare_parameter("free_distance_timeout", 0.5);
        const auto waypoints_file = declare_parameter("waypoints_file", std::string(use_sim_
            ? "/sim_ws/src/pure_pursuit/waypoints/aims.csv"
            : "/home/f1tenth/f1tenth_ws/src/waypoints/aims_closed.csv"));

        waypoints_ = pure_pursuit::ResamplePath(pure_pursuit::LoadCSV(waypoints_file), PATH_RESOLUTION);
        waypoints_in_execution_ = waypoints_;
        if (waypoints_.size() < 2) {
            RCLCPP_FATAL(get_logger(), "Could not load waypoints from %s", waypoints_file.c_str());
            throw std::runtime_error("no waypoints loaded from " + waypoints_file);
        }

        std::string odom_topic = "/odom";

        if(use_sim_){
         odom_topic = "ego_racecar/odom";
        }

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, 10, std::bind(&PurePursuit::OnOdom, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s, max_speed %.2f m/s",
                    waypoints_.size(), waypoints_file.c_str(), max_speed_);

        PublishWaypoints();

    }
    ~PurePursuit() {}


private:

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr free_distance_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr lookahead_pub_;
    std::vector<geometry_msgs::msg::Point> waypoints_;
    std::vector<geometry_msgs::msg::Point> waypoints_in_execution_;
    bool use_sim_;
    nav_msgs::msg::Odometry current_odom_;
    double free_distance_{std::numeric_limits<double>::infinity()};
    rclcpp::Time free_distance_stamp_{0, 0, RCL_ROS_TIME};

    double max_speed_, min_speed_, max_lat_accel_, max_decel_;
    double lookahead_min_, lookahead_max_, lookahead_time_min_, lookahead_time_max_, chord_tolerance_;
    double stop_margin_, free_distance_timeout_;

    static constexpr double WHEELBASE {0.3302};
    static constexpr double MAX_STEER {0.34};
    static constexpr double PATH_RESOLUTION {0.1};
    // laser is mounted this far forward of base_link (from ego_racecar.xacro)
    static constexpr double LASER_TO_BASE_LINK_X {0.275};


private:
    // Fastest speed from which every curvature limit along the next braking
    // distance can still be met: min over j of sqrt(v_curve_j^2 + 2 a s_j).
    double CurvatureSpeed(std::size_t start_idx, const std::vector<geometry_msgs::msg::Point>& waypoints) const
    {
        const double horizon = max_speed_ * max_speed_ / (2.0 * max_decel_) + 1.0;
        double allowed = max_speed_;
        double s = 0.0;
        std::size_t i = start_idx;
        const auto n = waypoints.size();
        for (std::size_t step = 0; step < n && s <= horizon; ++step) {
            const double kappa = pure_pursuit::CurvatureAt(i, waypoints);
            const double v_curve = std::min(max_speed_, std::sqrt(max_lat_accel_ / std::max(kappa, 1e-3)));
            allowed = std::min(allowed, std::sqrt(v_curve * v_curve + 2.0 * max_decel_ * s));
            const auto next = (i + 1) % n;
            s += pure_pursuit::Distance(waypoints[i], waypoints[next]);
            i = next;
        }
        return allowed;
    }

    // Longest lookahead whose chord stays within chord_tolerance of the path,
    // clamped to the speed-scaled window. Returns the goal waypoint index.
    std::size_t AdaptiveLookahead(const geometry_msgs::msg::Point& car, std::size_t start_idx, double speed,
                                  const std::vector<geometry_msgs::msg::Point>& waypoints) const
    {
        const double lo = std::min(std::max(lookahead_min_, lookahead_time_min_ * speed), lookahead_max_);
        const double hi = std::clamp(lookahead_time_max_ * speed, lo, lookahead_max_);
        const auto n = waypoints.size();

        std::vector<std::size_t> idx{start_idx};
        std::vector<double> arc{0.0};
        while (arc.back() < hi && idx.size() < n) {
            const auto next = (idx.back() + 1) % n;
            arc.push_back(arc.back() + pure_pursuit::Distance(waypoints[idx.back()], waypoints[next]));
            idx.push_back(next);
        }

        std::size_t goal = idx.back();
        for (std::size_t k = 1; k < idx.size(); ++k) {
            const auto& g = waypoints[idx[k]];
            const double cx = g.x - car.x, cy = g.y - car.y;
            const double chord = std::max(std::hypot(cx, cy), 1e-6);
            bool fits = true;
            for (std::size_t m = 1; m < k && fits; ++m) {
                const auto& p = waypoints[idx[m]];
                fits = std::abs(cx * (p.y - car.y) - cy * (p.x - car.x)) / chord <= chord_tolerance_;
            }
            if (!fits && arc[k] >= lo) {
                goal = idx[k - 1];
                break;
            }
        }
        return goal;
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
        const auto current_pose_waypoint_idx = pure_pursuit::GetIndexClosestToPoint(current_pose.position, waypoints);

        const double speed = std::max(0.0, current_odom_.twist.twist.linear.x);
        double target_speed = CurvatureSpeed(current_pose_waypoint_idx, waypoints);

        // Never outrun what RRT has checked: be able to stop stop_margin short
        // of the first blocked point. Ignored if RRT has gone quiet.
        bool limited_by_free_distance = false;
        if ((now() - free_distance_stamp_).seconds() < free_distance_timeout_) {
            const double room = std::max(0.0, free_distance_ - stop_margin_);
            const double v_free = std::sqrt(2.0 * max_decel_ * room);
            if (v_free < target_speed) {
                target_speed = v_free;
                limited_by_free_distance = true;
            }
        }

        const auto goal_waypoint_idx = AdaptiveLookahead(current_pose.position, current_pose_waypoint_idx,
                                                         std::max(speed, target_speed), waypoints);
        const auto &goal_point = waypoints[goal_waypoint_idx];
        const auto actual_lookahead_distance = std::hypot(goal_point.x - current_pose.position.x, goal_point.y - current_pose.position.y);

        PublishLookahead(goal_point);

        double yaw = tf2::getYaw(current_pose.orientation);

        const double dx = goal_point.x - current_pose.position.x;
        const double dy = goal_point.y - current_pose.position.y;
        const double goal_y_vehicle = -std::sin(yaw) * dx + std::cos(yaw) * dy;

        const auto curvature = 2 * goal_y_vehicle / std::max(actual_lookahead_distance * actual_lookahead_distance, 1e-6);

        const auto steering_angle = std::clamp(std::atan(curvature * WHEELBASE), -MAX_STEER, MAX_STEER);

        // The arc actually being commanded can be tighter than the path (while
        // recovering from tracking error), so respect its grip limit too.
        target_speed = std::min(target_speed, std::sqrt(max_lat_accel_ / std::max(std::abs(curvature), 1e-3)));
        if (!limited_by_free_distance) {
            target_speed = std::max(target_speed, min_speed_);
        }

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = steering_angle;
        drive_msg.drive.speed = target_speed;
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
        marker.pose.position.z = 0.0;  // z is the PathSource tag, not a height
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
        marker.color.a = 1.0;
        // red while tracking the reference, blue while on an RRT detour
        if (pure_pursuit::SourceOf(point) == pure_pursuit::kRrt) {
            marker.color.b = 1.0;
        } else {
            marker.color.r = 1.0;
        }
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
        current_odom_ = *odom_msg;
        if (use_sim_) {
            ExecutePurepursuit(odom_msg->pose.pose);
        }
    }

    void OnPose(const geometry_msgs::msg::PoseStamped::ConstPtr &pose_msg)
    {
        // The PF tracks the laser (it publishes map->laser); pure pursuit and
        // RRT's free distance are both measured from base_link.
        auto pose = pose_msg->pose;
        const double yaw = tf2::getYaw(pose.orientation);
        pose.position.x -= LASER_TO_BASE_LINK_X * std::cos(yaw);
        pose.position.y -= LASER_TO_BASE_LINK_X * std::sin(yaw);
        ExecutePurepursuit(pose);
    }

    void OnFreeDistance(const std_msgs::msg::Float32::ConstPtr &msg)
    {
        free_distance_ = msg->data;
        free_distance_stamp_ = now();
    }

    void OnPath(const nav_msgs::msg::Path::ConstPtr &path_msg)
    {
        if (path_msg->poses.empty()) {
            return;
        }

        // RRT publishes the whole execution loop, already spliced and
        // resampled, with each point's PathSource tag in z. Take it verbatim:
        // re-splicing or resampling here could drift from what RRT checked.
        std::vector<geometry_msgs::msg::Point> pts;
        pts.reserve(path_msg->poses.size());
        for (const auto &pose : path_msg->poses) {
            pts.push_back(pose.pose.position);
        }
        waypoints_in_execution_ = std::move(pts);
    }

};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}
