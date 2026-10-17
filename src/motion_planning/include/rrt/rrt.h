// RRT assignment

// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>
#include <optional>
#include <cstdint>
#include <fstream>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "pure_pursuit/waypoint_utils.hpp"

/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

// Struct defining the RRT_Node object in the RRT tree.
// More fields could be added to thiis struct if more info needed.
// You can choose to use this or not
typedef struct RRT_Node {
    double x, y;
    double cost; // only used for RRT*
    int parent; // index of parent node in the tree vector
    bool is_root = false;
    int current_index; // index of the current node in the tree vector
} RRT_Node;

class Point2D {
    public:
    Point2D(double x_val = 0.0, double y_val = 0.0) : x_(x_val), y_ (y_val) {}
    // Accessor methods for x and y

    double GetX() const { return x_; }
    double GetY() const { return y_; }

    private:
    double x_, y_;
};


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:
    constexpr static double OCCUPANCY_GRID_RES {0.05};
    constexpr static double OCCUPANCY_GRID_Y {3.0};
    constexpr static double OCCUPANCY_GRID_X {7.0};
    constexpr static int OCCUPANCY_GRID_WIDTH_CELLS {static_cast<int>(OCCUPANCY_GRID_X / OCCUPANCY_GRID_RES)};
    constexpr static int OCCUPANCY_GRID_HEIGHT_CELLS {static_cast<int>(OCCUPANCY_GRID_Y / OCCUPANCY_GRID_RES)};
    constexpr static int OCCUPANCY_GRID_TOTAL_CELLS {OCCUPANCY_GRID_WIDTH_CELLS * OCCUPANCY_GRID_HEIGHT_CELLS};
    constexpr static double OCCUPANCY_GRID_ORIGIN_X_WORLD {-0.5};
    constexpr static double OCCUPANCY_GRID_ORIGIN_Y_WORLD {-OCCUPANCY_GRID_Y/2.F};
    constexpr static double OCCUPANCY_GRID_BUFFER {0.3};
    constexpr static int OCCUPANCY_GRID_BUFFER_CELLS {static_cast<int>(OCCUPANCY_GRID_BUFFER / OCCUPANCY_GRID_RES)};
    constexpr static double INTERPOLATION_RESOLUTION {0.1};
    constexpr static int8_t OCCUPANCY_GRID_OCCUPIED {100};
    constexpr static double GOAL_THRESHOLD {0.1};
    constexpr static double MAX_EXPANSION_DIST {0.3};
    // laser is mounted this far forward of base_link (from ego_racecar.xacro)
    constexpr static double LASER_TO_BASE_LINK_X {0.275};

    //RRT*
    constexpr static double RRT_STAR_NEIGHBORHOOD_RADIUS {0.8};

    // TODO: add the publishers and subscribers you need
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_nav_pub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    bool use_sim_;

    // random generator, use this
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist_;
    std::uniform_real_distribution<> y_dist_;

    geometry_msgs::msg::Pose current_pose_;
    std::unique_ptr<nav_msgs::msg::OccupancyGrid> occupancy_grid_;
    std::vector<geometry_msgs::msg::Point> waypoints_;
    // Original waypoints with the latest RRT detour spliced in; used as the planning reference.
    std::vector<geometry_msgs::msg::Point> waypoints_in_execution_;
    float lookahead_distance_ {2.5F};

    // callbacks
    // ground-truth odometry from the simulator
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg);
    // particle-filter inferred pose on the real vehicle
    void pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg);
    // updates occupancy grid
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);
    void timer_callback();    // runs one RRT* plan from the current pose and publishes the result
    void RunRRT();

    // RRT methods
    Point2D sample();
    int nearest(const std::vector<RRT_Node> &tree, const Point2D &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, const Point2D &sampled_point);
    bool check_collision(const RRT_Node &nearest_node, const RRT_Node &new_node);
    bool is_goal(const RRT_Node &latest_added_node, double goal_x, double goal_y);
    std::vector<RRT_Node> find_path(std::vector<RRT_Node> &tree, const RRT_Node &latest_added_node);
    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, const RRT_Node &node);
    double line_cost(const RRT_Node &n1, const RRT_Node &n2);
    std::vector<int> near(const std::vector<RRT_Node> &tree, const RRT_Node &node);
    void UpdateBestParent(const std::vector<RRT_Node> &tree, const std::vector<int> &neighbours, RRT_Node &new_node);
    void Rewire(std::vector<RRT_Node> &tree, const std::vector<int> &neighbours, const RRT_Node &new_node);
    // Converts a found path (goal -> root) into root -> goal ordered points in the map frame.
    std::vector<geometry_msgs::msg::Point> PathToMapPoints(const std::vector<RRT_Node> &path) const;
    // Splices the RRT detour into a fresh copy of the original waypoints and stores it as the execution reference.
    void UpdateWaypointsInExecution(const std::vector<geometry_msgs::msg::Point> &rrt_path);

    // occupancy grid helpers
    void InitialiseOccupancyGrid();
    std::pair<double,double> GetCartesianCoordsFromPolarCoords(double r, double theta);
    // Converts world (base_link) coords to a flattened occupancy grid index.
    std::optional<uint32_t> WorldToMap(double x, double y);
    // Converts a flattened occupancy grid index to the world (base_link) coords of the cell center.
    std::optional<std::pair<double,double>> MapToWorld(uint32_t index);
    // Returns true if the given world (base_link) coords fall on an occupied cell.
    bool IsOccupied(double x, double y);

    // visualization helpers (all markers are published in the occupancy grid frame)
    void PublishWaypoints();
    void PublishTree(const std::vector<RRT_Node>& tree);
    void PublishGoal(double goal_x, double goal_y);
    void PublishPath(const std::vector<geometry_msgs::msg::Point>& path_points);


};

