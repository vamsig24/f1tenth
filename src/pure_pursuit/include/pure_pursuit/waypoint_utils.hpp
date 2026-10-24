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

// Where a point of the execution path came from, carried in Point::z (and so
// in the z of each pose of the /rrt_path nav_msgs/Path).
enum PathSource : int {
    kReference = 0,  // the loaded raceline
    kRrt = 1,        // an RRT detour spliced into it
};

inline PathSource SourceOf(const geometry_msgs::msg::Point& p) {
    return p.z > 0.5 ? kRrt : kReference;
}

// The waypoints form a closed loop: everything below walks past the end of the
// list back to the start, so nothing special happens at the lap line.

inline double Distance(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

// Index of the first waypoint at least `distance` of arc length ahead of start_idx.
inline std::size_t IndexAtDistanceLooped(double distance, std::size_t start_idx,
                                         const std::vector<geometry_msgs::msg::Point>& waypoints) {
    const auto n = waypoints.size();
    if (n < 2) {
        return start_idx;
    }
    double accum_dist = 0.0;
    std::size_t i = start_idx % n;
    for (std::size_t step = 1; step < n; ++step) {
        const auto next = (i + 1) % n;
        accum_dist += Distance(waypoints[i], waypoints[next]);
        i = next;
        if (accum_dist >= distance) {
            break;
        }
    }
    return i;
}

// Unsigned curvature (1/m) at waypoints[i] from the circle through the points
// about `half_span` metres of arc behind and ahead. Spanning a few points keeps
// the noise of a hand-logged or RRT path out of the estimate.
inline double CurvatureAt(std::size_t i, const std::vector<geometry_msgs::msg::Point>& waypoints,
                          double half_span = 0.3) {
    const auto n = waypoints.size();
    if (n < 3) {
        return 0.0;
    }
    std::size_t back = i % n;
    for (double d = 0.0; d < half_span;) {
        const auto prev = (back + n - 1) % n;
        d += Distance(waypoints[prev], waypoints[back]);
        back = prev;
    }
    const auto ahead = IndexAtDistanceLooped(half_span, i, waypoints);
    const auto& a = waypoints[back];
    const auto& b = waypoints[i % n];
    const auto& c = waypoints[ahead];
    const double ab = Distance(a, b), bc = Distance(b, c), ca = Distance(c, a);
    const double cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const double denom = ab * bc * ca;
    return denom < 1e-9 ? 0.0 : 2.0 * std::abs(cross) / denom;
}

// Resamples a polyline so consecutive points are spaced ~resolution metres apart.
// z is a PathSource tag, not a height, so it is never interpolated: a sample
// takes the tag of its segment, and a segment joining reference to RRT (the
// splice seam) counts as RRT since it is not on the reference.
inline std::vector<geometry_msgs::msg::Point> ResamplePath(const std::vector<geometry_msgs::msg::Point>& pts,
                                                           double resolution) {
    if (pts.size() < 2) {
        return pts;
    }
    std::vector<geometry_msgs::msg::Point> out{pts.front()};
    double carry = 0.0;  // arc already covered since the last emitted point
    for (std::size_t i = 1; i < pts.size(); ++i) {
        const double seg = Distance(pts[i - 1], pts[i]);
        if (seg < 1e-9) {
            continue;
        }
        double d = resolution - carry;
        for (; d <= seg; d += resolution) {
            geometry_msgs::msg::Point p;
            p.x = pts[i - 1].x + (pts[i].x - pts[i - 1].x) * d / seg;
            p.y = pts[i - 1].y + (pts[i].y - pts[i - 1].y) * d / seg;
            p.z = std::max(pts[i - 1].z, pts[i].z);
            out.push_back(p);
        }
        carry = seg - (d - resolution);
    }
    if (Distance(out.back(), pts.back()) > 1e-6) {
        out.push_back(pts.back());
    }
    return out;
}

// Replaces the stretch of `base` that `detour` bypasses with the detour itself,
// tagging the detour points kRrt. The result starts at the detour and runs once
// round the loop. A detour whose end projects behind its start (it would delete
// most of the lap) is rejected and `base` comes back unchanged.
//
// A detour may end beside the reference rather than on it (when the obstacle
// runs out of view). Its end offset is then faded out over `blend_length` of
// reference with a half-cosine, so the path never steps sideways at the seam;
// the blended points are tagged kRrt too, since they are not the reference.
inline std::vector<geometry_msgs::msg::Point> SpliceDetour(const std::vector<geometry_msgs::msg::Point>& base,
                                                           const std::vector<geometry_msgs::msg::Point>& detour,
                                                           double blend_length = 0.0) {
    const auto n = base.size();
    if (n < 2 || detour.size() < 2) {
        return base;
    }
    const auto start = GetIndexClosestToPoint(detour.front(), base);
    const auto end = GetIndexClosestToPoint(detour.back(), base);
    const auto span = (end + n - start) % n;
    if (span > n / 2) {
        return base;
    }
    std::vector<geometry_msgs::msg::Point> out(detour.begin(), detour.end());
    for (auto& p : out) {
        p.z = kRrt;
    }
    out.reserve(detour.size() + n);
    const double ex = detour.back().x - base[end].x;
    const double ey = detour.back().y - base[end].y;
    const bool blend = blend_length > 1e-6 && std::hypot(ex, ey) > 0.02;
    double arc = 0.0;
    for (std::size_t k = span + 1; k < n; ++k) {
        auto p = base[(start + k) % n];
        arc += Distance(base[(start + k - 1) % n], p);
        if (blend && arc < blend_length) {
            const double w = 0.5 * (1.0 + std::cos(M_PI * arc / blend_length));
            p.x += w * ex;
            p.y += w * ey;
            p.z = kRrt;
        }
        out.push_back(p);
    }
    return out;
}

}  // namespace pure_pursuit
