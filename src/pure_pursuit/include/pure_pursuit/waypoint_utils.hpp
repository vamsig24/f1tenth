#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>
#include "geometry_msgs/msg/point.hpp"

namespace pure_pursuit {

// Loads a waypoints CSV (single header line, "x,y" per row) into a list of points.
inline std::vector<geometry_msgs::msg::Point> LoadCSV(const std::string& file) {
    std::vector<geometry_msgs::msg::Point> points;
    std::ifstream f(file);
    std::string line;

    std::getline(f, line);  // header

    while (std::getline(f, line)) {
        std::stringstream ss(line);
        // std::cout << "Parsing line: " << ss.str() << std::endl;
        geometry_msgs::msg::Point p;
        char comma;
        ss >> p.x >> comma >> p.y;
        points.push_back(p);
    }

    return points;
}

// Returns the index of the waypoint closest to the given point.
inline std::size_t GetIndexClosestToPoint(const geometry_msgs::msg::Point& point,
                                          const std::vector<geometry_msgs::msg::Point>& waypoints) {
    const auto& min_it = std::min_element(waypoints.begin(), waypoints.end(),
        [&](const geometry_msgs::msg::Point& pose_1, const geometry_msgs::msg::Point& pose_2) {
            const auto distance_1 = std::hypot(point.x - pose_1.x, point.y - pose_1.y);
            const auto distance_2 = std::hypot(point.x - pose_2.x, point.y - pose_2.y);

            return distance_1 < distance_2;
        });

    return static_cast<std::size_t>(std::distance(waypoints.begin(), min_it));
}

// Returns the first waypoint index whose accumulated arc length from start_idx exceeds distance.
inline std::optional<size_t> GetClosestIndexAtDistance(double distance, std::size_t start_idx,
                                             const std::vector<geometry_msgs::msg::Point>& waypoints) {
    double accum_dist = 0;

    if (start_idx >= waypoints.size() - 1) {
        std::stringstream err;
        err << "Start index is greater than or equal to the size of current list of waypoints " << waypoints.size();
        return std::nullopt;
    }

    for (std::size_t i = start_idx + 1; i < waypoints.size(); i++) {
        const auto temp_dist = std::hypot(waypoints[i].x - waypoints[i - 1].x,
                                          waypoints[i].y - waypoints[i - 1].y);
        accum_dist += temp_dist;

        if (accum_dist > distance) {
            return i;
        }
    }

    return waypoints.size() - 1;
}

}  // namespace pure_pursuit
