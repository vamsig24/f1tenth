// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well
//
// Pipeline, every 50 ms:
//   1. Rebuild the inflated local grid from the latest scan.
//   2. Walk the execution path (reference + any spliced detour) out to the
//      planning horizon: stopping distance at the CURRENT speed plus a
//      reaction allowance. If a waypoint inside it is blocked, plan.
//   3. Goal = first free waypoint past every blocked stretch in view. If the
//      obstacle runs off the grid, take a free point beside the path at the
//      grid edge instead and keep extending as the car advances.
//   4. Goal-biased RRT, then shortcut + smooth, splice into the reference and
//      publish it for pure pursuit.
//   5. Publish how far ahead the execution path is clear so pure pursuit can
//      cap its speed to what has actually been checked.

#include "rrt/rrt.h"

#include <chrono>

// Destructor of the RRT class
RRT::~RRT() {
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// Constructor of the RRT class
RRT::RRT(): rclcpp::Node("rrt_node")
    ,occupancy_grid_pub_(this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1))
    ,gen((std::random_device())()) {

    use_sim_ = declare_parameter("use_sim", false);
    inflation_radius_ = declare_parameter("inflation_radius", 0.3);
    // braking measured on the car's bags is 3-5 m/s^2 (median ~4.2); 8 is a sim number
    max_decel_ = declare_parameter("max_decel", use_sim_ ? 8.0 : 4.0);
    reaction_time_ = declare_parameter("reaction_time", 0.3);
    horizon_margin_ = declare_parameter("horizon_margin", 1.0);
    min_horizon_ = declare_parameter("min_horizon", 1.5);
    goal_clearance_ = declare_parameter("goal_clearance", 0.6);
    max_lat_accel_ = declare_parameter("max_lat_accel", 8.0);
    path_margin_ = declare_parameter("path_margin", 0.1);
    max_lateral_offset_ = declare_parameter("max_lateral_offset", 1.5);
    base_frame_ = declare_parameter("base_frame", std::string(use_sim_ ? "ego_racecar/base_link" : "base_link"));
    pose_timeout_ = declare_parameter("pose_timeout", 0.5);
    const auto waypoints_file = declare_parameter("waypoints_file", std::string(use_sim_
        ? "/sim_ws/src/pure_pursuit/waypoints/aims.csv"
        : "/home/f1tenth/f1tenth_ws/src/waypoints/aims.csv"));

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

    InitialiseOccupancyGrid();
    clearance_.assign(OCCUPANCY_GRID_TOTAL_CELLS, 0.0F);

    waypoint_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "rrt_waypoints", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());
    tree_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("rrt_tree", 1);
    goal_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("rrt_goal", 1);
    // Latched, so a controller that starts late still gets the current detour.
    path_nav_pub_ = this->create_publisher<nav_msgs::msg::Path>(
        "rrt_path", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());
    free_distance_pub_ = this->create_publisher<std_msgs::msg::Float32>("/rrt/free_distance", 1);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50), std::bind(&RRT::timer_callback, this));

    // Same resampling as pure pursuit so both splice into identical references.
    waypoints_ = pure_pursuit::ResamplePath(pure_pursuit::LoadCSV(waypoints_file), PATH_RESOLUTION);
    waypoints_in_execution_ = waypoints_;
    if (waypoints_.size() < 2) {
        RCLCPP_FATAL(this->get_logger(), "Could not load waypoints from %s", waypoints_file.c_str());
        throw std::runtime_error("no waypoints loaded from " + waypoints_file);
    }
    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints from %s", waypoints_.size(), waypoints_file.c_str());

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
    // The sim publishes scans at ~250 Hz; inflating every one of them is wasted
    // work, so only cache it here and rebuild the grid at planning rate.
    latest_scan_ = scan_msg;
}

void RRT::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg) {
    // Ground-truth odometry from the simulator
    current_pose_ = odom_msg->pose.pose;
    current_speed_ = std::abs(odom_msg->twist.twist.linear.x);
    have_pose_ = true;
    last_pose_time_ = this->now();
}

void RRT::pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg) {
    // Particle-filter inferred pose on the real vehicle. The PF tracks the laser
    // (it publishes map->laser), but the grid and tree are rooted at base_link.
    auto pose = pose_msg->pose;
    const double yaw = tf2::getYaw(pose.orientation);
    pose.position.x -= LASER_TO_BASE_LINK_X * std::cos(yaw);
    pose.position.y -= LASER_TO_BASE_LINK_X * std::sin(yaw);

    // It carries no twist, so estimate speed from successive poses (low-passed
    // against PF jitter).
    const rclcpp::Time stamp(pose_msg->header.stamp);
    if (last_pose_position_.has_value()) {
        const double dt = (stamp - last_pose_stamp_).seconds();
        if (dt > 1e-3 && dt < 0.5) {
            const double v = pure_pursuit::Distance(pose.position, *last_pose_position_) / dt;
            current_speed_ = 0.7 * current_speed_ + 0.3 * v;
        }
    }
    last_pose_position_ = pose.position;
    last_pose_stamp_ = stamp;
    current_pose_ = pose;
    have_pose_ = true;
    last_pose_time_ = this->now();
}

void RRT::timer_callback() {
    RunRRT();
}

double RRT::PlanningHorizon() const {
    // Distance to react and then brake to a stop from the current speed. It
    // grows with speed on its own, so raising pure pursuit's max_speed needs
    // no retune here.
    const double v = current_speed_;
    const double horizon = v * reaction_time_ + v * v / (2.0 * max_decel_) + horizon_margin_;
    const double grid_reach = OCCUPANCY_GRID_ORIGIN_X_WORLD + OCCUPANCY_GRID_X - 0.5;
    return std::clamp(horizon, min_horizon_, grid_reach);
}

std::vector<PathSample> RRT::SamplePath(std::size_t start_idx) const {
    const double yaw = tf2::getYaw(current_pose_.orientation);
    const double c = std::cos(yaw), s = std::sin(yaw);
    const auto& path = waypoints_in_execution_;
    const auto n = path.size();
    const double max_arc = OCCUPANCY_GRID_X + OCCUPANCY_GRID_Y;

    std::vector<PathSample> samples;
    double arc = 0.0;
    std::size_t i = start_idx;
    for (std::size_t step = 0; step < n && arc <= max_arc; ++step) {
        const double dx = path[i].x - current_pose_.position.x;
        const double dy = path[i].y - current_pose_.position.y;
        PathSample p;
        p.x = c * dx + s * dy;
        p.y = -s * dx + c * dy;
        p.s = arc;
        p.in_grid = InGrid(p.x, p.y);
        p.occupied = p.in_grid && IsOccupied(p.x, p.y);
        samples.push_back(p);
        const auto next = (i + 1) % n;
        arc += pure_pursuit::Distance(path[i], path[next]);
        i = next;
    }
    return samples;
}

double RRT::FreeDistance(const std::vector<PathSample>& samples) const {
    // Arc length to the first blocked waypoint, or to where the path leaves
    // the grid (the limit of what has been checked). Waypoints within the
    // inflation radius of the car are skipped: the car is physically there.
    for (const auto& p : samples) {
        if (p.s < inflation_radius_) {
            continue;
        }
        if (!p.in_grid || p.occupied) {
            return p.s;
        }
    }
    return samples.empty() ? 0.0 : samples.back().s;
}

std::optional<Point2D> RRT::LateralFreePoint(const std::vector<PathSample>& samples, std::size_t j,
                                             double& offset) const {
    // Nearest free cell beside samples[j], along the path normal, either side.
    const std::size_t a = j > 0 ? j - 1 : 0;
    const std::size_t b = std::min(j + 1, samples.size() - 1);
    double tx = samples[b].x - samples[a].x, ty = samples[b].y - samples[a].y;
    const double norm = std::hypot(tx, ty);
    if (norm < 1e-6) {
        return std::nullopt;
    }
    tx /= norm;
    ty /= norm;
    for (double off = OCCUPANCY_GRID_RES; off <= max_lateral_offset_; off += OCCUPANCY_GRID_RES) {
        for (const double sign : {1.0, -1.0}) {
            const double gx = samples[j].x - sign * off * ty;
            const double gy = samples[j].y + sign * off * tx;
            if (gx > 0.3 && InGrid(gx, gy) && !IsOccupied(gx, gy)) {
                offset = off;
                return Point2D(gx, gy);
            }
        }
    }
    return std::nullopt;
}

double RRT::RejoinLength(double offset) const {
    // Length of a half-cosine lane change of lateral size `offset` whose peak
    // curvature, pi^2 * offset / (2 L^2), stays within max_lat_accel at the
    // current speed: L = pi * v * sqrt(offset / (2 a_lat)). Scales with speed,
    // so the rejoin stays drivable at full speed instead of forcing a brake.
    const double v = std::max(current_speed_, 1.0);
    return std::max(goal_clearance_, M_PI * v * std::sqrt(offset / (2.0 * max_lat_accel_)));
}

std::optional<Point2D> RRT::SelectGoal(const std::vector<PathSample>& samples, std::size_t first_blocked,
                                       double horizon, std::vector<Point2D>& goal_tail) {
    goal_tail.clear();
    auto goal_at = [&](std::size_t j) {
        const auto& p = samples[j];
        for (std::size_t k = j + 1; k < samples.size() && samples[k].s < p.s + goal_clearance_; ++k) {
            if (!samples[k].in_grid || samples[k].occupied) {
                break;
            }
            goal_tail.emplace_back(samples[k].x, samples[k].y);
        }
        return Point2D(p.x, p.y);
    };

    // The goal must be past every blocked stretch in view. Preferably far
    // enough beyond it that swinging back onto the reference is a gentle lane
    // change; if that runs off the grid, settle for the farthest free
    // waypoint past the obstacle that is in view.
    double s_preferred = horizon;
    double s_minimum = horizon;
    std::optional<std::size_t> fallback;
    std::size_t last_in_grid = first_blocked;
    for (std::size_t j = first_blocked; j < samples.size(); ++j) {
        const auto& p = samples[j];
        if (!p.in_grid) {
            break;
        }
        last_in_grid = j;
        if (p.occupied) {
            double offset = max_lateral_offset_;
            LateralFreePoint(samples, j, offset);
            s_preferred = std::max(s_preferred, p.s + RejoinLength(offset));
            s_minimum = std::max(s_minimum, p.s + goal_clearance_);
            fallback.reset();
            continue;
        }
        if (p.x > 0.3 && p.s >= s_minimum) {
            fallback = j;
            if (p.s >= s_preferred) {
                return goal_at(j);
            }
        }
    }
    if (fallback.has_value()) {
        return goal_at(*fallback);
    }

    // The obstacle itself runs past the edge of what we can see. Aim beside
    // the path at the grid edge, smallest offset first; the next cycles extend
    // the detour as more comes into view.
    double offset = 0.0;
    return LateralFreePoint(samples, last_in_grid, offset);
}

void RRT::RunRRT() {
    if (!have_pose_ || !latest_scan_ || waypoints_in_execution_.size() < 2) {
        return;
    }
    // The grid is live from the scan, but waypoints are projected with the pose;
    // a frozen pose puts them in the wrong place. Report zero free distance so
    // pure pursuit stops instead of falling back to an unchecked speed.
    const double pose_age = (this->now() - last_pose_time_).seconds();
    if (pose_age > pose_timeout_) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Localization pose is %.1fs old, not planning (is the particle filter alive?)", pose_age);
        std_msgs::msg::Float32 free_distance;
        free_distance.data = 0.0F;
        free_distance_pub_->publish(free_distance);
        return;
    }
    BuildOccupancyGrid();

    const auto current_pose_waypoint_idx = pure_pursuit::GetIndexClosestToPoint(current_pose_.position, waypoints_in_execution_);
    auto samples = SamplePath(current_pose_waypoint_idx);
    const double horizon = PlanningHorizon();

    // Trigger: a blocked waypoint within the horizon. Off-grid waypoints are
    // unknown, not blocked, so a turn leaving the grid does not fire it.
    std::optional<std::size_t> first_blocked;
    for (std::size_t j = 0; j < samples.size() && samples[j].s <= horizon; ++j) {
        if (samples[j].s >= inflation_radius_ && samples[j].occupied) {
            first_blocked = j;
            break;
        }
    }

    if (first_blocked.has_value()) {
        std::vector<Point2D> goal_tail;
        const auto goal = SelectGoal(samples, *first_blocked, horizon, goal_tail);
        if (!goal.has_value()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Path blocked and no free goal in view");
        } else {
            PublishGoal(goal->GetX(), goal->GetY());

            std::vector<RRT_Node> tree;
            tree.reserve(MAX_TREE_NODES);
            // Root the tree on the current execution path slightly ahead of the
            // car (where it will be once this plan lands), not at the car. The
            // new path then continues the one being tracked instead of
            // restarting from wherever tracking error has put the car, which
            // otherwise leaves a small S-bend that pure pursuit overshoots.
            const double root_lead = 0.2 + 0.1 * current_speed_;
            std::size_t root_idx = 0;
            for (std::size_t j = 0; j < *first_blocked; ++j) {
                if (samples[j].s >= root_lead) {
                    root_idx = j;
                    break;
                }
            }
            if (root_idx > 0 && samples[root_idx].occupied) {
                root_idx = 0;
            }
            std::vector<Point2D> prefix;
            for (std::size_t j = 0; j < root_idx; ++j) {
                prefix.emplace_back(samples[j].x, samples[j].y);
            }
            const double root_x = root_idx > 0 ? samples[root_idx].x : 0.0;
            const double root_y = root_idx > 0 ? samples[root_idx].y : 0.0;

            RRT_Node root_node;
            root_node.x = root_x;
            root_node.y = root_y;
            // Heading the car will have at the root: along the path it is
            // tracking, or its own heading if rooted at the car.
            root_node.heading = 0.0;
            if (root_idx > 0 && root_idx + 1 < samples.size()) {
                root_node.heading = std::atan2(samples[root_idx + 1].y - samples[root_idx].y,
                                               samples[root_idx + 1].x - samples[root_idx].x);
            }
            root_node.cost = 0.0;
            root_node.parent = -1;
            root_node.is_root = true;
            root_node.current_index = 0;
            tree.push_back(root_node);

            // Sample only the box that can matter: from the car to just past
            // the goal, with lateral room on both sides.
            const double y_lo = std::max(OCCUPANCY_GRID_ORIGIN_Y_WORLD, std::min(root_y, goal->GetY()) - 1.5);
            const double y_hi = std::min(OCCUPANCY_GRID_ORIGIN_Y_WORLD + OCCUPANCY_GRID_Y, std::max(root_y, goal->GetY()) + 1.5);
            const double x_hi = std::min(OCCUPANCY_GRID_ORIGIN_X_WORLD + OCCUPANCY_GRID_X, goal->GetX() + 1.0);
            std::uniform_real_distribution<> x_dist(std::max(0.0, root_x), x_hi);
            std::uniform_real_distribution<> y_dist(y_lo, y_hi);

            int iterations = 0;
            bool goal_reached = false;
            RRT_Node goal_node;
            while (static_cast<int>(tree.size()) < MAX_TREE_NODES && iterations++ < MAX_ITERATIONS) {
                const auto rand_point = sample(x_dist, y_dist, *goal);
                const auto nearest_index = nearest(tree, rand_point);
                auto new_node = steer(tree[nearest_index], rand_point);

                // Mostly forward: a detour that doubles back is never what we want.
                const auto& parent = tree[nearest_index];
                // Drivable: the edge may turn only as far as the car can at the
                // speed it could still have there after braking from the
                // current speed over the path so far. Steer toward the sample
                // but clamp the turn, so the tree still grows every iteration.
                // Near the car that allows a gentle swerve; farther out,
                // braking makes tighter turns possible, and pure pursuit's
                // curvature speed does that braking.
                const double step = line_cost(parent, new_node);
                const double v_sq = std::max(current_speed_ * current_speed_ - 2.0 * max_decel_ * parent.cost, 0.25);
                const double kappa_max = std::min(std::tan(MAX_STEER) / WHEELBASE, max_lat_accel_ / v_sq);
                const double max_turn = kappa_max * step;
                const double desired = std::atan2(new_node.y - parent.y, new_node.x - parent.x);
                const double turn = std::clamp(std::remainder(desired - parent.heading, 2.0 * M_PI), -max_turn, max_turn);
                new_node.heading = parent.heading + turn;
                new_node.x = parent.x + step * std::cos(new_node.heading);
                new_node.y = parent.y + step * std::sin(new_node.heading);
                // Mostly forward: a detour that doubles back is never what we want.
                if (new_node.x < parent.x - 0.05) {
                    continue;
                }
                if (check_collision(tree[nearest_index], new_node)) {
                    continue;
                }
                new_node.current_index = static_cast<int>(tree.size());
                tree.push_back(new_node);

                if (is_goal(new_node, goal->GetX(), goal->GetY())) {
                    goal_reached = true;
                    goal_node = new_node;
                    break;
                }
            }

            PublishTree(tree);

            if (goal_reached) {
                const auto path = find_path(tree, goal_node);
                std::vector<Point2D> raw;
                for (auto it = path.rbegin(); it != path.rend(); ++it) {
                    raw.emplace_back(it->x, it->y);
                }
                // The first edge sets the departure heading the tree was
                // constrained to, so shortcutting starts after it.
                auto detour = prefix;
                detour.push_back(raw.front());
                const auto shortcut = Shortcut(std::vector<Point2D>(raw.begin() + 1, raw.end()));
                detour.insert(detour.end(), shortcut.begin(), shortcut.end());
                // End on the reference tangent so the detour rejoins without a kink.
                if (!goal_tail.empty()) {
                    detour.push_back(goal_tail.back());
                }
                detour = SmoothPath(detour);
                const auto path_map_points = ToMapPoints(detour);
                RCLCPP_INFO(this->get_logger(),
                            "RRT detour: blocked at %.2f m (horizon %.2f m, v %.2f), goal (%.2f, %.2f), %d iterations",
                            samples[*first_blocked].s, horizon, current_speed_, goal->GetX(), goal->GetY(), iterations);
                UpdateWaypointsInExecution(path_map_points);
                PublishPath(waypoints_in_execution_);
                samples = SamplePath(pure_pursuit::GetIndexClosestToPoint(current_pose_.position, waypoints_in_execution_));
            } else {
                RCLCPP_WARN(this->get_logger(), "RRT could not find a path after %d iterations (tree size %zu)",
                            iterations, tree.size());
            }
        }
    }

    std_msgs::msg::Float32 free_distance;
    free_distance.data = static_cast<float>(FreeDistance(samples));
    free_distance_pub_->publish(free_distance);
    occupancy_grid_pub_->publish(*occupancy_grid_);
}

Point2D RRT::sample(std::uniform_real_distribution<>& x_dist, std::uniform_real_distribution<>& y_dist,
                    const Point2D& goal) {
    // Goal bias: without it, landing within GOAL_THRESHOLD of a goal several
    // metres away by uniform sampling alone is rare.
    std::uniform_real_distribution<> unit(0.0, 1.0);
    if (unit(gen) < GOAL_SAMPLE_RATE) {
        return goal;
    }
    return { x_dist(gen), y_dist(gen)};
}


int RRT::nearest(const std::vector<RRT_Node> &tree, const Point2D &sampled_point) {
    // This method returns the nearest node on the tree to the sampled point
    const auto& min_it = std::min_element(tree.begin(), tree.end(),
    [&](const RRT_Node& node_1, const RRT_Node& node_2) {
        const auto distance_1 = std::hypot(sampled_point.GetX() - node_1.x, sampled_point.GetY() - node_1.y);
        const auto distance_2 = std::hypot(sampled_point.GetX() - node_2.x, sampled_point.GetY() - node_2.y);

        return distance_1 < distance_2;
    });

    return static_cast<int>(std::distance(tree.begin(), min_it));
}

RRT_Node RRT::steer(RRT_Node &nearest_node, const Point2D &sampled_point) {
    // Expand the tree towards the sample point, at most MAX_EXPANSION_DIST.
    RRT_Node new_node;

    const double dx = sampled_point.GetX() - nearest_node.x;
    const double dy = sampled_point.GetY() - nearest_node.y;
    const double dist = std::hypot(dx, dy);

    if (dist <= MAX_EXPANSION_DIST) {
        new_node.x = sampled_point.GetX();
        new_node.y = sampled_point.GetY();
    } else {
        const double scale = MAX_EXPANSION_DIST / dist;
        new_node.x = nearest_node.x + scale * dx;
        new_node.y = nearest_node.y + scale * dy;
    }

    new_node.parent = nearest_node.current_index;
    new_node.is_root = false;
    new_node.current_index = -1;
    new_node.cost = nearest_node.cost + line_cost(nearest_node, new_node);

    return new_node;
}

bool RRT::check_collision(const RRT_Node &nearest_node, const RRT_Node &new_node) {
    // true if the straight segment between the two nodes is in collision
    return !SegmentFree(Point2D(nearest_node.x, nearest_node.y), Point2D(new_node.x, new_node.y));
}

double RRT::SegmentClearance(const Point2D& a, const Point2D& b) const {
    // Smallest obstacle clearance along the segment. Points within
    // inflation_radius of the car are skipped: the car is physically there,
    // so if it is already inside the margin of something the way out must
    // still be plannable.
    const auto length = std::hypot(b.GetX() - a.GetX(), b.GetY() - a.GetY());
    const auto steps = std::max(1, static_cast<int>(std::ceil(length / INTERPOLATION_RESOLUTION)));
    double lowest = std::numeric_limits<double>::infinity();
    for (int step = 0; step <= steps; ++step) {
        const double t = static_cast<double>(step) / steps;
        const auto x = a.GetX() + t * (b.GetX() - a.GetX());
        const auto y = a.GetY() + t * (b.GetY() - a.GetY());
        if (std::hypot(x, y) < inflation_radius_) {
            continue;
        }
        lowest = std::min(lowest, Clearance(x, y));
    }
    return lowest;
}

bool RRT::SegmentFree(const Point2D& a, const Point2D& b, double extra_margin) const {
    return SegmentClearance(a, b) >= inflation_radius_ + extra_margin;
}

bool RRT::is_goal(const RRT_Node &latest_added_node, double goal_x, double goal_y) {
    const auto distance_to_goal = std::hypot(latest_added_node.x - goal_x, latest_added_node.y - goal_y);
    return distance_to_goal < GOAL_THRESHOLD;
}

std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, const RRT_Node &latest_added_node) {
    // Walks parents from the goal node back to the root (goal -> root order).
    std::vector<RRT_Node> found_path;
    found_path.reserve(tree.size());
    auto node_found = latest_added_node;

    while (!node_found.is_root) {
        found_path.push_back(node_found);
        node_found = tree[node_found.parent];
    }

    found_path.push_back(node_found);
    return found_path;
}

std::vector<Point2D> RRT::Shortcut(const std::vector<Point2D>& path) {
    // Greedy line-of-sight pruning: from each kept point jump to the farthest
    // point still visible. Removes the random zig-zag RRT leaves behind.
    if (path.size() < 3) {
        return path;
    }
    // Only shortcuts that keep path_margin beyond the inflation are taken, so
    // the result does not graze obstacles where there is room not to; in a
    // tight gap it keeps the RRT's own edges.
    std::vector<Point2D> out{path.front()};
    std::size_t i = 0;
    while (i + 1 < path.size()) {
        std::size_t j = path.size() - 1;
        while (j > i + 1 && !SegmentFree(path[i], path[j], path_margin_)) {
            --j;
        }
        out.push_back(path[j]);
        i = j;
    }
    return out;
}

std::vector<Point2D> RRT::SmoothPath(const std::vector<Point2D>& path) {
    // Chaikin corner cutting on the sparse shortcut vertices (endpoints
    // pinned). Each pass cuts a quarter of every segment, so corners round out
    // over metres rather than the few centimetres a dense path would give.
    // Backs off to fewer passes if rounding brings the path closer to an
    // obstacle than inflation + path_margin (or than the unsmoothed path
    // already was, in a gap tighter than that).
    double raw_clearance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        raw_clearance = std::min(raw_clearance, SegmentClearance(path[i], path[i + 1]));
    }
    const double required = std::min(inflation_radius_ + path_margin_, raw_clearance);
    for (int passes = 5; passes >= 0; --passes) {
        auto smooth = path;
        for (int iter = 0; iter < passes && smooth.size() > 2; ++iter) {
            std::vector<Point2D> next{smooth.front()};
            for (std::size_t i = 0; i + 1 < smooth.size(); ++i) {
                const auto& a = smooth[i];
                const auto& b = smooth[i + 1];
                if (i > 0) {
                    next.emplace_back(0.75 * a.GetX() + 0.25 * b.GetX(), 0.75 * a.GetY() + 0.25 * b.GetY());
                }
                if (i + 2 < smooth.size()) {
                    next.emplace_back(0.25 * a.GetX() + 0.75 * b.GetX(), 0.25 * a.GetY() + 0.75 * b.GetY());
                }
            }
            next.push_back(smooth.back());
            smooth = std::move(next);
        }
        bool free = true;
        for (std::size_t i = 0; i + 1 < smooth.size() && free; ++i) {
            free = SegmentClearance(smooth[i], smooth[i + 1]) >= required;
        }
        if (free || passes == 0) {
            std::vector<geometry_msgs::msg::Point> pts;
            for (const auto& p : smooth) {
                geometry_msgs::msg::Point q;
                q.x = p.GetX();
                q.y = p.GetY();
                pts.push_back(q);
            }
            std::vector<Point2D> out;
            for (const auto& q : pure_pursuit::ResamplePath(pts, PATH_RESOLUTION)) {
                out.emplace_back(q.x, q.y);
            }
            return out;
        }
    }
    return path;
}

// RRT* methods
double RRT::cost(std::vector<RRT_Node> &tree, const RRT_Node &node) {
    (void)tree;
    return node.cost;
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
    return std::hypot(n1.x - n2.x, n1.y - n2.y);
}

std::vector<int> RRT::near(const std::vector<RRT_Node> &tree,   const RRT_Node &node) {
    std::vector<int> neighborhood;
    for(size_t i =0; i < tree.size(); ++i){
        const auto distance = std::hypot(tree[i].x - node.x, tree[i].y - node.y);
        if (distance <= RRT_STAR_NEIGHBORHOOD_RADIUS) {
            neighborhood.push_back(i);
        }
    }
    return neighborhood;
}

void RRT::InitialiseOccupancyGrid() {
    if (!occupancy_grid_) {
        occupancy_grid_ = std::make_unique<nav_msgs::msg::OccupancyGrid>();
        occupancy_grid_->info.resolution = OCCUPANCY_GRID_RES;
        occupancy_grid_->info.width = OCCUPANCY_GRID_WIDTH_CELLS;
        occupancy_grid_->info.height = OCCUPANCY_GRID_HEIGHT_CELLS;
        occupancy_grid_->info.origin.position.x = OCCUPANCY_GRID_ORIGIN_X_WORLD;
        occupancy_grid_->info.origin.position.y = OCCUPANCY_GRID_ORIGIN_Y_WORLD;
        occupancy_grid_->info.origin.orientation.w = 1.0;
        occupancy_grid_->header.frame_id = base_frame_;
    }
    occupancy_grid_->data.assign(OCCUPANCY_GRID_TOTAL_CELLS, 0);
}

void RRT::BuildOccupancyGrid() {
    // Distance from every cell to the nearest scan return (two-pass chamfer),
    // so planning can ask for more clearance than the bare inflation where
    // there is room. Cells closer than inflation_radius (car half-width plus
    // margin) are occupied, so the planner can treat the car as a point.
    InitialiseOccupancyGrid();
    constexpr float kFar = 1e3F;
    clearance_.assign(OCCUPANCY_GRID_TOTAL_CELLS, kFar);
    const auto& scan = *latest_scan_;
    for (size_t i = 0; i < scan.ranges.size(); i++) {
        const auto range = scan.ranges[i];
        if (!std::isfinite(range) || range < scan.range_min || range > scan.range_max) {
            continue;
        }
        const auto angle = scan.angle_min + i * scan.angle_increment;
        auto [x, y] = GetCartesianCoordsFromPolarCoords(range, angle);
        x += LASER_TO_BASE_LINK_X;

        // the lidar sees parts of the car (cables, mounts); inflated, these block
        // the path right ahead and collapse the free distance pure pursuit drives to
        if (x > FOOTPRINT_MIN_X && x < FOOTPRINT_MAX_X && std::abs(y) < FOOTPRINT_HALF_WIDTH) {
            continue;
        }

        const auto index = WorldToMap(x, y);
        if (index.has_value()) {
            clearance_[index.value()] = 0.0F;
        }
    }

    const int w = OCCUPANCY_GRID_WIDTH_CELLS, h = OCCUPANCY_GRID_HEIGHT_CELLS;
    const float d1 = OCCUPANCY_GRID_RES, d2 = static_cast<float>(OCCUPANCY_GRID_RES * M_SQRT2);
    auto relax = [&](int r, int c, int dr, int dc, float cost) {
        const int rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < h && cc >= 0 && cc < w) {
            clearance_[r * w + c] = std::min(clearance_[r * w + c], clearance_[rr * w + cc] + cost);
        }
    };
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            relax(r, c, 0, -1, d1); relax(r, c, -1, 0, d1); relax(r, c, -1, -1, d2); relax(r, c, -1, 1, d2);
        }
    }
    for (int r = h - 1; r >= 0; --r) {
        for (int c = w - 1; c >= 0; --c) {
            relax(r, c, 0, 1, d1); relax(r, c, 1, 0, d1); relax(r, c, 1, 1, d2); relax(r, c, 1, -1, d2);
        }
    }
    for (int k = 0; k < OCCUPANCY_GRID_TOTAL_CELLS; ++k) {
        if (clearance_[k] < inflation_radius_) {
            occupancy_grid_->data[k] = OCCUPANCY_GRID_OCCUPIED;
        }
    }
    occupancy_grid_->header.stamp = scan.header.stamp;
}

std::pair <double,double> RRT::GetCartesianCoordsFromPolarCoords(double r, double theta) {
    double x = r * cos(theta);
    double y = r * sin(theta);

    return std::make_pair(x, y);
}

std::optional<uint32_t> RRT::WorldToMap(double x, double y) {
    const int col = static_cast<int>(std::floor((x - OCCUPANCY_GRID_ORIGIN_X_WORLD) / OCCUPANCY_GRID_RES));
    const int row = static_cast<int>(std::floor((y - OCCUPANCY_GRID_ORIGIN_Y_WORLD) / OCCUPANCY_GRID_RES));
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

bool RRT::InGrid(double x, double y) const {
    return x >= OCCUPANCY_GRID_ORIGIN_X_WORLD && x < OCCUPANCY_GRID_ORIGIN_X_WORLD + OCCUPANCY_GRID_X &&
           y >= OCCUPANCY_GRID_ORIGIN_Y_WORLD && y < OCCUPANCY_GRID_ORIGIN_Y_WORLD + OCCUPANCY_GRID_Y;
}

double RRT::Clearance(double x, double y) const {
    if (!InGrid(x, y)) {
        return 0.0;
    }
    const int col = static_cast<int>((x - OCCUPANCY_GRID_ORIGIN_X_WORLD) / OCCUPANCY_GRID_RES);
    const int row = static_cast<int>((y - OCCUPANCY_GRID_ORIGIN_Y_WORLD) / OCCUPANCY_GRID_RES);
    return clearance_[row * OCCUPANCY_GRID_WIDTH_CELLS + col];
}

bool RRT::IsOccupied(double x, double y) const {
    return Clearance(x, y) < inflation_radius_;
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

std::vector<geometry_msgs::msg::Point> RRT::ToMapPoints(const std::vector<Point2D>& path) const {
    std::vector<geometry_msgs::msg::Point> points;
    points.reserve(path.size());

    const double yaw = tf2::getYaw(current_pose_.orientation);
    const double origin_x = current_pose_.position.x;
    const double origin_y = current_pose_.position.y;

    for (const auto& p : path) {
        geometry_msgs::msg::Point point;
        point.x = origin_x + std::cos(yaw) * p.GetX() - std::sin(yaw) * p.GetY();
        point.y = origin_y + std::sin(yaw) * p.GetX() + std::cos(yaw) * p.GetY();
        points.push_back(point);
    }

    return points;
}

void RRT::UpdateWaypointsInExecution(const std::vector<geometry_msgs::msg::Point>& rrt_path) {
    // Rebuild from the untouched original, replacing the spanned segment with
    // the RRT detour. Resampled so a jump at the splice seam is interpolated
    // and shows up as blocked if it cuts through an obstacle.
    // A detour that ends beside the reference fades back onto it over the
    // same drivable lane-change length used to place goals.
    const auto end_idx = pure_pursuit::GetIndexClosestToPoint(rrt_path.back(), waypoints_);
    const double end_offset = pure_pursuit::Distance(rrt_path.back(), waypoints_[end_idx]);
    const double blend = end_offset > 0.02 ? RejoinLength(end_offset) : 0.0;
    waypoints_in_execution_ = pure_pursuit::ResamplePath(pure_pursuit::SpliceDetour(waypoints_, rrt_path, blend), PATH_RESOLUTION);
}

void RRT::PublishPath(const std::vector<geometry_msgs::msg::Point>& path_points) {
    // The whole execution loop (reference with the detour spliced in) in the
    // map frame, already resampled to PATH_RESOLUTION. Each pose's z is the
    // pure_pursuit::PathSource tag, so the controller tracks it as-is with no
    // splicing or resampling of its own.
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
