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
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "pure_pursuit/waypoint_utils.hpp"

using namespace std;

// Struct defining the RRT_Node object in the RRT tree.
typedef struct RRT_Node {
    double x, y;
    double cost; // only used for RRT*
    int parent; // index of parent node in the tree vector
    bool is_root = false;
    int current_index; // index of the current node in the tree vector
    double heading = 0.0; // direction of the edge from the parent (base_link)
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

// One reference waypoint as seen from the car, for the trigger and goal search.
struct PathSample {
    double x, y;   // base_link frame
    double s;      // arc length from the car's closest waypoint
    bool in_grid;
    bool occupied;
};


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:
    // Local grid in base_link. 9.5 m ahead covers the stopping distance plus
    // reaction distance up to ~7 m/s at max_decel 5; beyond that the free
    // distance cap in pure pursuit holds the car to what the grid can see.
    constexpr static double OCCUPANCY_GRID_RES {0.05};
    constexpr static double OCCUPANCY_GRID_Y {6.0};
    constexpr static double OCCUPANCY_GRID_X {10.0};
    constexpr static int OCCUPANCY_GRID_WIDTH_CELLS {static_cast<int>(OCCUPANCY_GRID_X / OCCUPANCY_GRID_RES)};
    constexpr static int OCCUPANCY_GRID_HEIGHT_CELLS {static_cast<int>(OCCUPANCY_GRID_Y / OCCUPANCY_GRID_RES)};
    constexpr static int OCCUPANCY_GRID_TOTAL_CELLS {OCCUPANCY_GRID_WIDTH_CELLS * OCCUPANCY_GRID_HEIGHT_CELLS};
    constexpr static double OCCUPANCY_GRID_ORIGIN_X_WORLD {-0.5};
    constexpr static double OCCUPANCY_GRID_ORIGIN_Y_WORLD {-OCCUPANCY_GRID_Y/2.F};
    constexpr static double INTERPOLATION_RESOLUTION {0.05};
    constexpr static int8_t OCCUPANCY_GRID_OCCUPIED {100};
    constexpr static double GOAL_THRESHOLD {0.25};
    constexpr static double MAX_EXPANSION_DIST {0.3};
    constexpr static int MAX_TREE_NODES {800};
    constexpr static int MAX_ITERATIONS {4000};
    constexpr static double GOAL_SAMPLE_RATE {0.15};
    constexpr static double PATH_RESOLUTION {0.1};
    // laser is mounted this far forward of base_link (from ego_racecar.xacro)
    constexpr static double LASER_TO_BASE_LINK_X {0.275};
    // vehicle geometry, same as pure pursuit
    constexpr static double WHEELBASE {0.3302};
    constexpr static double MAX_STEER {0.34};

    //RRT*
    constexpr static double RRT_STAR_NEIGHBORHOOD_RADIUS {0.8};

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_nav_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr free_distance_pub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    bool use_sim_;

    // random generator, use this
    std::mt19937 gen;

    geometry_msgs::msg::Pose current_pose_;
    bool have_pose_ {false};
    double current_speed_ {0.0};
    // pose-delta speed estimate for the real car, where the pose carries no twist
    std::optional<geometry_msgs::msg::Point> last_pose_position_;
    rclcpp::Time last_pose_stamp_;

    sensor_msgs::msg::LaserScan::ConstSharedPtr latest_scan_;
    std::unique_ptr<nav_msgs::msg::OccupancyGrid> occupancy_grid_;
    // metres from each grid cell to the nearest scan return
    std::vector<float> clearance_;
    std::vector<geometry_msgs::msg::Point> waypoints_;
    // Original waypoints with the latest RRT detour spliced in; used as the planning reference.
    std::vector<geometry_msgs::msg::Point> waypoints_in_execution_;

    // Tunables (ROS params). None depend on max speed: the horizon is derived
    // from the measured speed and the braking model.
    double inflation_radius_;     // car half-width + margin
    double max_decel_;            // shared meaning with pure pursuit
    double reaction_time_;        // plan + actuation latency budget
    double horizon_margin_;       // extra room past the stopping distance
    double min_horizon_;
    double goal_clearance_;       // minimum distance past the blocked stretch for the goal
    double max_lat_accel_;        // shared meaning with pure pursuit; sizes the rejoin
    double path_margin_;          // clearance beyond the inflation that shortcut/smoothing keep when there is room
    double max_lateral_offset_;   // search width when the obstacle runs off the grid

    // callbacks
    // ground-truth odometry from the simulator
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg);
    // particle-filter inferred pose on the real vehicle
    void pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg);
    // caches the scan; the grid is rebuilt at planning rate in RunRRT
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);
    void timer_callback();    // runs one RRT plan from the current pose and publishes the result
    void RunRRT();

    // trigger + goal selection
    double PlanningHorizon() const;
    std::vector<PathSample> SamplePath(std::size_t start_idx) const;
    std::optional<Point2D> SelectGoal(const std::vector<PathSample>& samples, std::size_t first_blocked,
                                      double horizon, std::vector<Point2D>& goal_tail);
    double FreeDistance(const std::vector<PathSample>& samples) const;
    std::optional<Point2D> LateralFreePoint(const std::vector<PathSample>& samples, std::size_t j,
                                            double& offset) const;
    double RejoinLength(double offset) const;

    // RRT methods
    Point2D sample(std::uniform_real_distribution<>& x_dist, std::uniform_real_distribution<>& y_dist,
                   const Point2D& goal);
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

    // path post-processing (base_link frame)
    double SegmentClearance(const Point2D& a, const Point2D& b) const;
    bool SegmentFree(const Point2D& a, const Point2D& b, double extra_margin = 0.0) const;
    std::vector<Point2D> Shortcut(const std::vector<Point2D>& path);
    std::vector<Point2D> SmoothPath(const std::vector<Point2D>& path);
    // Converts root -> goal points in base_link to the map frame.
    std::vector<geometry_msgs::msg::Point> ToMapPoints(const std::vector<Point2D> &path) const;
    // Splices the RRT detour into a fresh copy of the original waypoints and stores it as the execution reference.
    void UpdateWaypointsInExecution(const std::vector<geometry_msgs::msg::Point> &rrt_path);

    // occupancy grid helpers
    void InitialiseOccupancyGrid();
    void BuildOccupancyGrid();
    std::pair<double,double> GetCartesianCoordsFromPolarCoords(double r, double theta);
    // Converts world (base_link) coords to a flattened occupancy grid index.
    std::optional<uint32_t> WorldToMap(double x, double y);
    // Converts a flattened occupancy grid index to the world (base_link) coords of the cell center.
    std::optional<std::pair<double,double>> MapToWorld(uint32_t index);
    // Returns true if the given world (base_link) coords fall on an occupied cell.
    // Off-grid counts as occupied: the planner must not route through the unknown.
    bool IsOccupied(double x, double y) const;
    // Metres to the nearest scan return; 0 off the grid.
    double Clearance(double x, double y) const;
    bool InGrid(double x, double y) const;

    // visualization helpers (all markers are published in the occupancy grid frame)
    void PublishWaypoints();
    void PublishTree(const std::vector<RRT_Node>& tree);
    void PublishGoal(double goal_x, double goal_y);
    // Publishes the full tagged execution path (see PublishPath in rrt.cpp).
    void PublishPath(const std::vector<geometry_msgs::msg::Point>& path_points);


};
