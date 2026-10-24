// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "rrt/rrt.h"

#include <chrono>

// Destructor of the RRT class
RRT::~RRT() {
    // Do something in here, free up used memory, print message, etc.
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// Constructor of the RRT class
RRT::RRT(): rclcpp::Node("rrt_node"), gen((std::random_device())())
    ,occupancy_grid_pub_(this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1)),
occupancy_grid_(std::make_unique<nav_msgs::msg::OccupancyGrid>()),
    x_dist_ {0, OCCUPANCY_GRID_ORIGIN_X_WORLD + OCCUPANCY_GRID_X},
    y_dist_ {OCCUPANCY_GRID_ORIGIN_Y_WORLD, OCCUPANCY_GRID_ORIGIN_Y_WORLD + OCCUPANCY_GRID_Y} {

    // ROS publishers
    // TODO: create publishers for the the drive topic, and other topics you might need

    // ROS subscribers
    // TODO: create subscribers as you need
    use_sim_ = false;

    string scan_topic = "/scan";
    if (use_sim_) {
        string odom_topic = "ego_racecar/odom";
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
          odom_topic, 1, std::bind(&RRT::odom_callback, this, std::placeholders::_1));
    } else {
        string pose_topic = "/pf/viz/inferred_pose";
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          pose_topic, 1, std::bind(&RRT::pose_callback, this, std::placeholders::_1));
    }
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, 1, std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    // TODO: create a occupancy grid
    
    InitialiseOccupancyGrid();

    waypoint_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "rrt_waypoints", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());
    tree_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("rrt_tree", 1);
    goal_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("rrt_goal", 1);
    path_nav_pub_ = this->create_publisher<nav_msgs::msg::Path>("rrt_path", 1);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50), std::bind(&RRT::timer_callback, this));

    waypoints_ = pure_pursuit::LoadCSV("/home/f1tenth_ws/src/pure_pursuit/waypoints/aims.csv");
    waypoints_in_execution_ = waypoints_;

        std::cout << "Loaded waypoints " << waypoints_.size() << " from CSV" << std::endl;

        // RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints", waypoints_.size());

        PublishWaypoints();


    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");
}

void RRT::PublishWaypoints()
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

void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // The scan callback, update your occupancy grid here
    // Args:
    //    scan_msg (*LaserScan): pointer to the incoming scan message
    // Returns:
    //

    // TODO: update your occupancy grid
    InitialiseOccupancyGrid();

    for(size_t i =0 ; i< scan_msg->ranges.size(); i++) {
        const auto range = scan_msg->ranges[i];
        if (!std::isfinite(range) || range < scan_msg->range_min || range > scan_msg->range_max) {
            continue;
        }
        const auto angle = scan_msg->angle_min + i * scan_msg->angle_increment;
        auto [x, y] = GetCartesianCoordsFromPolarCoords(range, angle);
        x += LASER_TO_BASE_LINK_X;

        const auto index = WorldToMap(x, y);
        if (!index.has_value()) {
            continue;
        }

        const auto center_col = static_cast<int>(index.value() % OCCUPANCY_GRID_WIDTH_CELLS);
        const auto center_row = static_cast<int>(index.value() / OCCUPANCY_GRID_WIDTH_CELLS);
        const auto col_start = std::max(0, center_col - OCCUPANCY_GRID_BUFFER_CELLS);
        const auto col_end = std::min(OCCUPANCY_GRID_WIDTH_CELLS - 1, center_col + OCCUPANCY_GRID_BUFFER_CELLS);
        const auto row_start = std::max(0, center_row - OCCUPANCY_GRID_BUFFER_CELLS);
        const auto row_end = std::min(OCCUPANCY_GRID_HEIGHT_CELLS - 1, center_row + OCCUPANCY_GRID_BUFFER_CELLS);

        for (int r = row_start; r <= row_end; ++r) {
            for (int c = col_start; c <= col_end; ++c) {
                occupancy_grid_->data[r * OCCUPANCY_GRID_WIDTH_CELLS + c] = OCCUPANCY_GRID_OCCUPIED;
            }
        }

    }

    occupancy_grid_pub_->publish(*occupancy_grid_);


}

void RRT::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg) {
    // Ground-truth odometry from the simulator
    current_pose_ = odom_msg->pose.pose;
}

void RRT::pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg) {
    // Particle-filter inferred pose on the real vehicle
    current_pose_ = pose_msg->pose;
}

void RRT::timer_callback() {
    RunRRT();
}

void RRT::RunRRT() {
    // The RRT main loop happens here, using the latest pose and occupancy grid.
    // tree as std::vector
    std::vector<RRT_Node> tree;
    tree.reserve(100);

    RRT_Node root_node;
    root_node.x = 0.0;
    root_node.y = 0.0;
    root_node.cost = 0.0;
    root_node.parent = -1;
    root_node.is_root = true;
    root_node.current_index = 0;
    tree.push_back(root_node);

    const auto current_pose_waypoint_idx = pure_pursuit::GetIndexClosestToPoint(current_pose_.position, waypoints_in_execution_);

     std::optional<size_t> goal_waypoint_idx_opt = pure_pursuit::GetClosestIndexAtDistance(lookahead_distance_, current_pose_waypoint_idx, waypoints_in_execution_);

     if(!goal_waypoint_idx_opt.has_value()) {
         // No valid goal waypoint found at the specified lookahead distance.
         return;
     }
     const auto goal_waypoint_idx = goal_waypoint_idx_opt.value();

     double yaw = tf2::getYaw(current_pose_.orientation);

     double goal_x_vehicle_collision_free = 0.0;
     double goal_y_vehicle_collision_free = 0.0;

    auto to_vehicle_frame = [&](const geometry_msgs::msg::Point &point) {
        const double dx = point.x - current_pose_.position.x;
        const double dy = point.y - current_pose_.position.y;
        return std::make_pair(std::cos(yaw) * dx + std::sin(yaw) * dy,
                              -std::sin(yaw) * dx + std::cos(yaw) * dy);
    };

    // Only plan when a reference waypoint between the car and the lookahead
    // goal lands on an occupied cell; a clear stretch lets pure pursuit follow it.
    bool corridor_blocked = false;
    for(size_t i = current_pose_waypoint_idx; i <= goal_waypoint_idx; ++i) {
        const auto [wx, wy] = to_vehicle_frame(waypoints_in_execution_[i]);
        if(IsOccupied(wx, wy)) {
            corridor_blocked = true;
            break;
        }
    }

    if(!corridor_blocked) {
        return;
    }

    for(size_t i = goal_waypoint_idx; i < waypoints_in_execution_.size(); ++i) {

        const auto [goal_x_vehicle, goal_y_vehicle] = to_vehicle_frame(waypoints_in_execution_[i]);

        if(!IsOccupied(goal_x_vehicle, goal_y_vehicle)) {
            goal_x_vehicle_collision_free = goal_x_vehicle;
            goal_y_vehicle_collision_free = goal_y_vehicle;
            break;
        }
    }

    if(std::abs(goal_x_vehicle_collision_free) < 1e-6 && std::abs(goal_y_vehicle_collision_free) < 1e-6) {
        // RCLCPP_WARN(this->get_logger(), "No collision-free goal found");
        return;
    }
   
    RCLCPP_INFO(this->get_logger(), "RRT triggered: corridor blocked, planning to goal (%.2f, %.2f)",
                goal_x_vehicle_collision_free, goal_y_vehicle_collision_free);

    PublishGoal(goal_x_vehicle_collision_free, goal_y_vehicle_collision_free);

    int iterations = 0;
    constexpr int MAX_ITERATIONS = 1000;
    bool goal_reached = false;
    RRT_Node goal_node;

    // const auto updated_lookahead_distance = std::hypot(goal_x_vehicle_collision_free.x, goal_y_vehicle_collision_free.y);


    while (tree.size() < 100 && iterations++ < MAX_ITERATIONS) {
        const auto rand_point = sample();
        const auto nearest_index = nearest(tree, rand_point);
        auto new_node = steer(tree[nearest_index], rand_point);

        if (new_node.x < tree[nearest_index].x) {
            continue;  
        }

        if (check_collision(tree[nearest_index], new_node)) {
            continue;
        }

        new_node.current_index = static_cast<int>(tree.size());

        // const auto neighbours = near(tree, new_node);
        // UpdateBestParent(tree, neighbours, new_node);
        tree.push_back(new_node);
        // Rewire(tree, neighbours, tree.back());

        if (is_goal(new_node,  goal_x_vehicle_collision_free,  goal_y_vehicle_collision_free)) {
            goal_reached = true;
            goal_node = new_node;
            break;
        }
    }


    PublishTree(tree);

    if (goal_reached) {
        const auto path = find_path(tree, goal_node);
        const auto path_map_points = PathToMapPoints(path);
        RCLCPP_INFO(this->get_logger(), "RRT found a path with %zu nodes in %d iterations",
                    path.size(), iterations);
        PublishPath(path_map_points);
        UpdateWaypointsInExecution(path_map_points);
    } else {
        RCLCPP_WARN(this->get_logger(), "RRT could not find a path after %d iterations (tree size %zu)",
                    iterations, tree.size());
    }
}

Point2D RRT::sample() {
    // This method returns a sampled point from the free space
    // You should restrict so that it only samples a small region
    // of interest around the car's current position
    // Args:
    // Returns:
    //     sampled_point (Point2D): the sampled point in free space

    // Point2D sampled_point;
 

    return { x_dist_(gen), y_dist_(gen)};
    // TODO: fill in this method
    // look up the documentation on how to use std::mt19937 devices with a distribution
    // the generator and the distribution is created for you (check the header file)

    // return sampled_point;
}


int RRT::nearest(const std::vector<RRT_Node> &tree, const Point2D &sampled_point) {
    // This method returns the nearest node on the tree to the sampled point
    // Args:
    //     tree (std::vector<RRT_Node>): the current RRT tree
    //     sampled_point (Point2D): the sampled point in free space
    // Returns:
    //     nearest_node (int): index of nearest node on the tree


    const auto& min_it = std::min_element(tree.begin(), tree.end(),
    [&](const RRT_Node& node_1, const RRT_Node& node_2) {
        const auto distance_1 = std::hypot(sampled_point.GetX() - node_1.x, sampled_point.GetY() - node_1.y);
        const auto distance_2 = std::hypot(sampled_point.GetX() - node_2.x, sampled_point.GetY() - node_2.y);

        return distance_1 < distance_2;
    });
    
    return static_cast<int>(std::distance(tree.begin(), min_it));
    // TODO: fill in this method

    // return nearest_node;
}

RRT_Node RRT::steer(RRT_Node &nearest_node, const Point2D &sampled_point) {
    // The function steer:(x,y)->z returns a point such that z is “closer” 
    // to y than x is. The point z returned by the function steer will be 
    // such that z minimizes ||z−y|| while at the same time maintaining 
    //||z−x|| <= max_expansion_dist, for a prespecified max_expansion_dist > 0

    // basically, expand the tree towards the sample point (within a max dist)

    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    sampled_point (Point2D): the sampled point in free space
    // Returns:
    //    new_node (RRT_Node): new node created from steering

    RRT_Node new_node;

    const double dx = sampled_point.GetX() - nearest_node.x;
    const double dy = sampled_point.GetY() - nearest_node.y;
    const double dist = std::hypot(dx, dy);

    if (dist <= MAX_EXPANSION_DIST) {
        // sample is already reachable: z = y
        new_node.x = sampled_point.GetX();
        new_node.y = sampled_point.GetY();
    } else {
        // move exactly MAX_EXPANSION_DIST toward the sample
        const double scale = MAX_EXPANSION_DIST / dist;
        new_node.x = nearest_node.x + scale * dx;
        new_node.y = nearest_node.y + scale * dy;
    }

    new_node.parent = nearest_node.current_index;
    new_node.is_root = false;
    new_node.current_index = -1;
    new_node.cost = nearest_node.cost + line_cost(nearest_node, new_node);
    // TODO: fill in this method

    return new_node;
}

bool RRT::check_collision(const RRT_Node &nearest_node, const RRT_Node &new_node) {
    // This method returns a boolean indicating if the path between the 
    // nearest node and the new node created from steering is collision free
    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    new_node (RRT_Node): new node created from steering
    // Returns:
    //    collision (bool): true if in collision, false otherwise


    const auto distance_between_nodes =  std::hypot(new_node.x - nearest_node.x, new_node.y - nearest_node.y);
    const auto total_number_of_steps = std::max(1, static_cast<int>(distance_between_nodes / INTERPOLATION_RESOLUTION));

    for( size_t step = 0; step <= total_number_of_steps; ++step) {
        const double t = static_cast<double>(step) / total_number_of_steps;
        const auto x = nearest_node.x + t * (new_node.x - nearest_node.x);
        const auto y = nearest_node.y + t * (new_node.y - nearest_node.y);

        if (IsOccupied(x, y)) {
            return true;
        }
    }

    // based on the interpolation, we also need to check the final new node position
    if (IsOccupied(new_node.x, new_node.y)) {
        return true;
    }

    return false;
}

bool RRT::is_goal(const RRT_Node &latest_added_node, double goal_x, double goal_y) {
    // This method checks if the latest node added to the tree is close
    // enough (defined by goal_threshold) to the goal so we can terminate
    // the search and find a path
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree
    //   goal_x (double): x coordinate of the current goal
    //   goal_y (double): y coordinate of the current goal
    // Returns:
    //   close_enough (bool): true if node close enough to the goal

    const auto distance_to_goal = std::hypot(latest_added_node.x - goal_x, latest_added_node.y - goal_y);

    return distance_to_goal < GOAL_THRESHOLD;
    // TODO: fill in this method

}

std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, const RRT_Node &latest_added_node) {
    // This method traverses the tree from the node that has been determined
    // as goal
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree that has been
    //      determined to be close enough to the goal
    // Returns:
    //   path (std::vector<RRT_Node>): the vector that represents the order of
    //      of the nodes traversed as the found path
    
    std::vector<RRT_Node> found_path;
    found_path.reserve(tree.size());
    auto node_found = latest_added_node;

    while (!node_found.is_root) {
        found_path.push_back(node_found);
        node_found = tree[node_found.parent];
    }

    found_path.push_back(node_found);

    // TODO: fill in this method

    return found_path;
}

// RRT* methods
double RRT::cost(std::vector<RRT_Node> &tree, const RRT_Node &node) {
    // This method returns the cost associated with a node
    // Args:
    //    tree (std::vector<RRT_Node>): the current tree
    //    node (RRT_Node): the node the cost is calculated for
    // Returns:
    //    cost (double): the cost value associated with the node

    double cost = 0;
    // TODO: fill in this method

    return cost;
}

void RRT::UpdateBestParent(const std::vector<RRT_Node> &tree, const std::vector<int> &neighbours, RRT_Node &new_node) {

    for(const auto idx : neighbours){
        const auto &neighbour = tree[idx];
        const auto candidate_cost = neighbour.cost + line_cost(neighbour, new_node);
        if(candidate_cost < new_node.cost && !check_collision(neighbour, new_node)){
            new_node.parent = neighbour.current_index;
            new_node.cost = candidate_cost;
        }
    }
}

void RRT::Rewire(std::vector<RRT_Node> &tree, const std::vector<int> &neighbours, const RRT_Node &new_node) {
    for(const auto idx : neighbours){
        auto &neighbour = tree[idx];
        if(neighbour.is_root){
            continue;
        }
        const auto candidate_cost = new_node.cost + line_cost(new_node, neighbour);
        if(candidate_cost < neighbour.cost && !check_collision(new_node, neighbour)){
            neighbour.parent = new_node.current_index;
            neighbour.cost = candidate_cost;
        }
    }
}


double RRT::line_cost(const RRT_Node &n1, const RRT_Node &n2) {
    // This method returns the cost of the straight line path between two nodes
    // Args:
    //    n1 (RRT_Node): the RRT_Node at one end of the path
    //    n2 (RRT_Node): the RRT_Node at the other end of the path
    // Returns:
    //    cost (double): the cost value associated with the path

    return std::hypot(n1.x - n2.x, n1.y - n2.y);
    // TODO: fill in this method
}

std::vector<int> RRT::near(const std::vector<RRT_Node> &tree,   const RRT_Node &node) {
    // This method returns the set of Nodes in the neighborhood of a 
    // node.
    // Args:
    //   tree (std::vector<RRT_Node>): the current tree
    //   node (RRT_Node): the node to find the neighborhood for
    // Returns:
    //   neighborhood (std::vector<int>): the index of the nodes in the neighborhood

    std::vector<int> neighborhood;
    for(size_t i =0; i < tree.size(); ++i){
        const auto distance = std::hypot(tree[i].x - node.x, tree[i].y - node.y);
        if (distance <= RRT_STAR_NEIGHBORHOOD_RADIUS) {
            neighborhood.push_back(i);
        }
    }
    // TODO:: fill in this method

    return neighborhood;
}

void RRT::InitialiseOccupancyGrid() {
    occupancy_grid_ = std::make_unique<nav_msgs::msg::OccupancyGrid>();
    occupancy_grid_->info.resolution = OCCUPANCY_GRID_RES;
    occupancy_grid_->info.width = OCCUPANCY_GRID_WIDTH_CELLS;
    occupancy_grid_->info.height = OCCUPANCY_GRID_HEIGHT_CELLS;
    occupancy_grid_->data.resize(OCCUPANCY_GRID_TOTAL_CELLS, 0);
    occupancy_grid_->info.origin.position.x = OCCUPANCY_GRID_ORIGIN_X_WORLD;
    occupancy_grid_->info.origin.position.y = OCCUPANCY_GRID_ORIGIN_Y_WORLD;
    occupancy_grid_->header.frame_id = "ego_racecar/base_link";
    // TODO: fill in this method
}

std::pair <double,double> RRT::GetCartesianCoordsFromPolarCoords(double r, double theta) {
    double x = r * cos(theta);
    double y = r * sin(theta);

    return std::make_pair(x, y);
}

std::optional<uint32_t> RRT::WorldToMap(double x, double y) {
    const int col = static_cast<int>((x - OCCUPANCY_GRID_ORIGIN_X_WORLD) / OCCUPANCY_GRID_RES);
    const int row = static_cast<int>((y - OCCUPANCY_GRID_ORIGIN_Y_WORLD) / OCCUPANCY_GRID_RES);
    if (col < 0 || col >= OCCUPANCY_GRID_WIDTH_CELLS ||
        row < 0 || row >= OCCUPANCY_GRID_HEIGHT_CELLS) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(row) * OCCUPANCY_GRID_WIDTH_CELLS + static_cast<uint32_t>(col);
}

std::optional<std::pair<double,double>> RRT::MapToWorld(uint32_t index) {
    if (index >= static_cast<uint32_t>(OCCUPANCY_GRID_TOTAL_CELLS)) {
        return std::nullopt;
    }
    const uint32_t col = index % OCCUPANCY_GRID_WIDTH_CELLS;
    const uint32_t row = index / OCCUPANCY_GRID_WIDTH_CELLS;
    const double x = OCCUPANCY_GRID_ORIGIN_X_WORLD + (col + 0.5) * OCCUPANCY_GRID_RES;
    const double y = OCCUPANCY_GRID_ORIGIN_Y_WORLD + (row + 0.5) * OCCUPANCY_GRID_RES;
    return std::make_pair(x, y);
}

bool RRT::IsOccupied(double x, double y) {
    const auto index = WorldToMap(x, y);
    if (!index.has_value()) {
        return true;
    }
    return occupancy_grid_->data[index.value()] == OCCUPANCY_GRID_OCCUPIED;
}

void RRT::PublishTree(const std::vector<RRT_Node>& tree) {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker vertices;
    vertices.header.frame_id = occupancy_grid_->header.frame_id;
    vertices.header.stamp = now();
    vertices.ns = "rrt_vertices";
    vertices.id = 0;
    vertices.type = visualization_msgs::msg::Marker::POINTS;
    vertices.action = visualization_msgs::msg::Marker::ADD;
    vertices.scale.x = 0.05;
    vertices.scale.y = 0.05;
    vertices.color.a = 1.0;
    vertices.color.b = 1.0;

    visualization_msgs::msg::Marker edges;
    edges.header.frame_id = occupancy_grid_->header.frame_id;
    edges.header.stamp = now();
    edges.ns = "rrt_edges";
    edges.id = 1;
    edges.type = visualization_msgs::msg::Marker::LINE_LIST;
    edges.action = visualization_msgs::msg::Marker::ADD;
    edges.scale.x = 0.02;
    edges.color.a = 1.0;
    edges.color.r = 1.0;

    for (const auto& node : tree) {
        geometry_msgs::msg::Point point;
        point.x = node.x;
        point.y = node.y;
        vertices.points.push_back(point);

        if (!node.is_root && node.parent >= 0) {
            geometry_msgs::msg::Point parent_point;
            parent_point.x = tree[node.parent].x;
            parent_point.y = tree[node.parent].y;
            edges.points.push_back(point);
            edges.points.push_back(parent_point);
        }
    }

    marker_array.markers.push_back(vertices);
    marker_array.markers.push_back(edges);
    tree_pub_->publish(marker_array);
}

void RRT::PublishGoal(double goal_x, double goal_y) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = occupancy_grid_->header.frame_id;
    marker.header.stamp = now();
    marker.ns = "rrt_goal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = goal_x;
    marker.pose.position.y = goal_y;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.a = 1.0;
    marker.color.b = 1.0;
    goal_pub_->publish(marker);
}
std::vector<geometry_msgs::msg::Point> RRT::PathToMapPoints(const std::vector<RRT_Node>& path) const {
    std::vector<geometry_msgs::msg::Point> points;
    points.reserve(path.size());

    const double yaw = tf2::getYaw(current_pose_.orientation);
    const double origin_x = current_pose_.position.x;
    const double origin_y = current_pose_.position.y;

    // find_path returns goal -> root; reverse so points run root -> goal
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        geometry_msgs::msg::Point point;
        point.x = origin_x + std::cos(yaw) * it->x - std::sin(yaw) * it->y;
        point.y = origin_y + std::sin(yaw) * it->x + std::cos(yaw) * it->y;
        points.push_back(point);
    }

    return points;
}

void RRT::UpdateWaypointsInExecution(const std::vector<geometry_msgs::msg::Point>& rrt_path) {
    if (rrt_path.empty()) {
        return;
    }

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

void RRT::PublishPath(const std::vector<geometry_msgs::msg::Point>& path_points) {
    // nav_msgs/Path in the map frame so the controller can track it as reference waypoints
    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = "map";
    path_msg.header.stamp = now();

    for (const auto& point : path_points) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path_msg.header;
        pose.pose.position = point;
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);
    }

    path_nav_pub_->publish(path_msg);
}