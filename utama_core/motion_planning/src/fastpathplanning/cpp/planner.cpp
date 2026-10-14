#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <optional>
#include <unordered_map>
#include <limits>

namespace py = pybind11;

constexpr double EPS = 1e-9;

// ============================================================================
// Core Geometric Helpers (identical to numba_kernels.py formulas)
// ============================================================================

inline double distance_point_to_segment(double px, double py, double sx, double sy, double ex, double ey) {
    double seg_dx = ex - sx;
    double seg_dy = ey - sy;
    double pt_dx = px - sx;
    double pt_dy = py - sy;
    double seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy;
    if (seg_len_sq < EPS) {
        return std::hypot(px - sx, py - sy);
    }
    double t = (pt_dx * seg_dx + pt_dy * seg_dy) / seg_len_sq;
    double cx, cy;
    if (t < 0.0) {
        cx = sx;
        cy = sy;
    } else if (t > 1.0) {
        cx = ex;
        cy = ey;
    } else {
        cx = sx + t * seg_dx;
        cy = sy + t * seg_dy;
    }
    return std::hypot(px - cx, py - cy);
}

inline std::pair<double, double> closest_point_on_segment_xy(
    double px, double py, double sx, double sy, double ex, double ey
) {
    double seg_dx = ex - sx;
    double seg_dy = ey - sy;
    double seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy;
    if (seg_len_sq < EPS) {
        return {sx, sy};
    }
    double t = ((px - sx) * seg_dx + (py - sy) * seg_dy) / seg_len_sq;
    if (t < 0.0) {
        return {sx, sy};
    } else if (t > 1.0) {
        return {ex, ey};
    }
    return {sx + t * seg_dx, sy + t * seg_dy};
}

inline int point_orientation(double p1x, double p1y, double p2x, double p2y, double p3x, double p3y) {
    double val = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x);
    if (std::abs(val) < EPS) return 0;
    return (val < 0.0) ? 1 : 2;
}

inline bool on_segment(double px, double py, double qx, double qy, double rx, double ry) {
    return std::min(px, rx) - EPS <= qx && qx <= std::max(px, rx) + EPS &&
           std::min(py, ry) - EPS <= qy && qy <= std::max(py, ry) + EPS;
}

inline bool segments_intersect(
    double p1x, double p1y, double q1x, double q1y,
    double p2x, double p2y, double q2x, double q2y
) {
    int o1 = point_orientation(p1x, p1y, q1x, q1y, p2x, p2y);
    int o2 = point_orientation(p1x, p1y, q1x, q1y, q2x, q2y);
    int o3 = point_orientation(p2x, p2y, q2x, q2y, p1x, p1y);
    int o4 = point_orientation(p2x, p2y, q2x, q2y, q1x, q1y);

    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && on_segment(p1x, p1y, p2x, p2y, q1x, q1y)) return true;
    if (o2 == 0 && on_segment(p1x, p1y, q2x, q2y, q1x, q1y)) return true;
    if (o3 == 0 && on_segment(p2x, p2y, p1x, p1y, q2x, q2y)) return true;
    if (o4 == 0 && on_segment(p2x, p2y, q1x, q1y, q2x, q2y)) return true;
    return false;
}

inline std::pair<double, double> distance_between_segments(
    double ax0, double ay0, double ax1, double ay1,
    double bx0, double by0, double bx1, double by1
) {
    double d3 = distance_point_to_segment(bx0, by0, ax0, ay0, ax1, ay1);
    if (segments_intersect(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1)) {
        return {0.0, d3};
    }
    double d1 = distance_point_to_segment(ax0, ay0, bx0, by0, bx1, by1);
    double d2 = distance_point_to_segment(ax1, ay1, bx0, by0, bx1, by1);
    double d4 = distance_point_to_segment(bx1, by1, ax0, ay0, ax1, ay1);
    return {std::min({d1, d2, d3, d4}), d3};
}

// ============================================================================
// Obstacle Scanners & Kernels
// ============================================================================

inline std::tuple<int, double, double> scan_collides(
    double seg_x0, double seg_y0, double seg_x1, double seg_y1,
    double clearance,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n,
    int sticky_idx
) {
    double seg_min_x = std::min(seg_x0, seg_x1) - clearance;
    double seg_max_x = std::max(seg_x0, seg_x1) + clearance;
    double seg_min_y = std::min(seg_y0, seg_y1) - clearance;
    double seg_max_y = std::max(seg_y0, seg_y1) + clearance;

    int closest_idx = -1;
    double min_dist_to_robot = 1e18;
    double sticky_dist_to_robot = -1.0;

    for (size_t i = 0; i < n; ++i) {
        double ox0i = ox0[i], oy0i = oy0[i], ox1i = ox1[i], oy1i = oy1[i];
        double o_min_x = std::min(ox0i, ox1i);
        double o_max_x = std::max(ox0i, ox1i);
        if (o_max_x < seg_min_x || o_min_x > seg_max_x) continue;
        double o_min_y = std::min(oy0i, oy1i);
        double o_max_y = std::max(oy0i, oy1i);
        if (o_max_y < seg_min_y || o_min_y > seg_max_y) continue;

        auto [dist_between_segs, dist_to_robot] = distance_between_segments(
            ox0i, oy0i, ox1i, oy1i, seg_x0, seg_y0, seg_x1, seg_y1
        );
        if (dist_between_segs < clearance) {
            if (static_cast<int>(i) == sticky_idx) {
                sticky_dist_to_robot = dist_to_robot;
            }
            if (dist_to_robot < min_dist_to_robot) {
                min_dist_to_robot = dist_to_robot;
                closest_idx = static_cast<int>(i);
            }
        }
    }
    return {closest_idx, min_dist_to_robot, sticky_dist_to_robot};
}

inline std::tuple<int, bool, double, double> collides(
    double ax, double ay, double bx, double by,
    double clearance,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n,
    int sticky_idx, double switch_margin
) {
    auto [closest_idx, min_dist, sticky_dist] = scan_collides(
        ax, ay, bx, by, clearance, ox0, oy0, ox1, oy1, n, sticky_idx
    );
    if (closest_idx < 0) {
        return {-1, false, 0.0, 0.0};
    }
    int idx = closest_idx;
    bool from_sticky = false;
    if (sticky_idx >= 0 && sticky_dist >= 0.0 &&
        !(ox0[closest_idx] == ox0[sticky_idx] && oy0[closest_idx] == oy0[sticky_idx] &&
          ox1[closest_idx] == ox1[sticky_idx] && oy1[closest_idx] == oy1[sticky_idx]) &&
        sticky_dist <= min_dist + switch_margin) {
        idx = sticky_idx;
        from_sticky = true;
    }

    double cx = ox0[idx], cy = oy0[idx], dx = ox1[idx], dy = oy1[idx];
    double denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
    if (std::abs(denom) >= EPS) {
        double t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom;
        double u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom;
        if (-EPS <= t && t <= 1.0 + EPS && -EPS <= u && u <= 1.0 + EPS) {
            return {idx, from_sticky, ax + t * (bx - ax), ay + t * (by - ay)};
        }
    }

    auto [pcx, pcy] = closest_point_on_segment_xy(ax, ay, cx, cy, dx, dy);
    auto [pdx, pdy] = closest_point_on_segment_xy(bx, by, cx, cy, dx, dy);
    double best_x = cx, best_y = cy;
    double best = distance_point_to_segment(cx, cy, ax, ay, bx, by);
    double d = distance_point_to_segment(dx, dy, ax, ay, bx, by);
    if (d < best) {
        best = d;
        best_x = dx;
        best_y = dy;
    }
    d = std::hypot(pcx - ax, pcy - ay);
    if (d < best) {
        best = d;
        best_x = pcx;
        best_y = pcy;
    }
    d = std::hypot(pdx - bx, pdy - by);
    if (d < best) {
        best = d;
        best_x = pdx;
        best_y = pdy;
    }
    return {idx, from_sticky, best_x, best_y};
}

inline std::tuple<bool, double, double, int, bool> find_subgoal(
    double opx, double opy, double ux, double uy,
    double subgoal_distance, int multiple, double clearance,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n,
    bool has_rect, double rmin_x, double rmax_x, double rmin_y, double rmax_y
) {
    int last_hit = -1;
    bool rect_last = false;
    while (multiple <= 10) {
        double sub_x = opx + subgoal_distance * ux * multiple;
        double sub_y = opy + subgoal_distance * uy * multiple;
        if (has_rect && rmin_x < sub_x && sub_x < rmax_x && rmin_y < sub_y && sub_y < rmax_y) {
            multiple += 1;
            rect_last = true;
            last_hit = -1;
            continue;
        }
        double box_min_x = sub_x - clearance;
        double box_max_x = sub_x + clearance;
        double box_min_y = sub_y - clearance;
        double box_max_y = sub_y + clearance;
        int hit = -1;
        for (size_t i = 0; i < n; ++i) {
            double ox0i = ox0[i], oy0i = oy0[i], ox1i = ox1[i], oy1i = oy1[i];
            if (std::max(ox0i, ox1i) < box_min_x || std::min(ox0i, ox1i) > box_max_x) continue;
            if (std::max(oy0i, oy1i) < box_min_y || std::min(oy0i, oy1i) > box_max_y) continue;
            if (distance_point_to_segment(sub_x, sub_y, ox0i, oy0i, ox1i, oy1i) < clearance) {
                hit = static_cast<int>(i);
                break;
            }
        }
        if (hit >= 0) {
            multiple += 1;
            rect_last = false;
            last_hit = hit;
            continue;
        }
        return {true, sub_x, sub_y, -1, false};
    }
    return {false, 0.0, 0.0, last_hit, rect_last};
}

inline double clamp_to_obstacle_clearance(
    double ox, double oy, double ux, double uy, double max_distance, double clearance,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n
) {
    double clamped = max_distance;
    for (size_t i = 0; i < n; ++i) {
        if (distance_point_to_segment(ox + ux * clamped, oy + uy * clamped, ox0[i], oy0[i], ox1[i], oy1[i]) >= clearance) {
            continue;
        }
        double lo = 0.0, hi = clamped;
        for (int step = 0; step < 12; ++step) {
            double mid = (lo + hi) / 2.0;
            if (distance_point_to_segment(ox + ux * mid, oy + uy * mid, ox0[i], oy0[i], ox1[i], oy1[i]) >= clearance) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        clamped = std::min(clamped, lo);
    }
    return clamped;
}

inline std::pair<double, double> sanitize_target(
    double tx, double ty, double rx, double ry, double clearance,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n,
    const double* exempt, size_t n_exempt
) {
    std::vector<bool> repelling(n, true);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n_exempt; ++j) {
            if (ox0[i] == exempt[j * 4 + 0] && oy0[i] == exempt[j * 4 + 1] &&
                ox1[i] == exempt[j * 4 + 2] && oy1[i] == exempt[j * 4 + 3]) {
                repelling[i] = false;
                break;
            }
        }
    }
    double push = clearance * 1.05;
    for (int pass = 0; pass < 5; ++pass) {
        bool collision_found = false;
        for (size_t i = 0; i < n; ++i) {
            if (!repelling[i]) continue;
            if (distance_point_to_segment(tx, ty, ox0[i], oy0[i], ox1[i], oy1[i]) < clearance) {
                auto [cx, cy] = closest_point_on_segment_xy(tx, ty, ox0[i], oy0[i], ox1[i], oy1[i]);
                double px = tx - cx;
                double py = ty - cy;
                double norm = std::hypot(px, py);
                if (norm == 0.0) {
                    px = rx - cx;
                    py = ry - cy;
                    norm = std::hypot(px, py);
                }
                tx = cx + (px / norm) * push;
                ty = cy + (py / norm) * push;
                collision_found = true;
            }
        }
        if (!collision_found) break;
    }
    return {tx, ty};
}

// ============================================================================
// check_segment and smooth_path C++ implementations
// ============================================================================

struct Point {
    double x, y;
};

struct Segment {
    Point a, b;
};

inline double round_to(double val, int precision) {
    double factor = std::pow(10.0, precision);
    return std::round(val * factor) / factor;
}

void check_segment_recursive(
    Point seg_a, Point seg_b,
    int recursion_length, int max_recursion_length,
    Point target,
    double fb_min_x, double fb_max_x, double fb_min_y, double fb_max_y,
    int robot_id,
    double clearance, double subgoal_distance,
    bool has_rect, double rmin_x, double rmax_x, double rmin_y, double rmax_y,
    double detour_switch_margin, double moving_ghost_min_m, int precision,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n,
    const bool* is_static,
    py::dict last_detour_side,
    py::dict last_obstacle,
    std::vector<Segment>& out_trajectory
) {
    double seg_len = std::hypot(seg_b.x - seg_a.x, seg_b.y - seg_a.y);
    int sticky_idx = -1;
    py::tuple obstacle_mem_key;
    if (robot_id >= 0) {
        obstacle_mem_key = py::make_tuple(robot_id, recursion_length);
        if (last_obstacle.contains(obstacle_mem_key)) {
            py::object obs_seg_obj = last_obstacle[obstacle_mem_key];
            if (py::isinstance<py::tuple>(obs_seg_obj)) {
                py::tuple t = obs_seg_obj.cast<py::tuple>();
                if (t.size() == 2) {
                    py::sequence pt_a = t[0].cast<py::sequence>();
                    py::sequence pt_b = t[1].cast<py::sequence>();
                    double sx0 = pt_a[0].cast<double>();
                    double sy0 = pt_a[1].cast<double>();
                    double sx1 = pt_b[0].cast<double>();
                    double sy1 = pt_b[1].cast<double>();
                    for (size_t i = 0; i < n; ++i) {
                        if (ox0[i] == sx0 && oy0[i] == sy0 && ox1[i] == sx1 && oy1[i] == sy1) {
                            sticky_idx = static_cast<int>(i);
                            break;
                        }
                    }
                }
            }
        }
    }

    auto [closest_idx, from_sticky, px, py] = collides(
        seg_a.x, seg_a.y, seg_b.x, seg_b.y,
        clearance, ox0, oy0, ox1, oy1, n,
        sticky_idx, detour_switch_margin
    );

    if (closest_idx < 0 || recursion_length >= max_recursion_length) {
        if (robot_id >= 0 && last_obstacle.contains(obstacle_mem_key)) {
            py::dict(last_obstacle).attr("pop")(obstacle_mem_key, py::none());
        }
        out_trajectory.push_back({seg_a, seg_b});
        return;
    }

    // Save sticky obstacle segment into last_obstacle
    if (robot_id >= 0) {
        py::array_t<double> pt0(2);
        pt0.mutable_at(0) = ox0[closest_idx];
        pt0.mutable_at(1) = oy0[closest_idx];
        py::array_t<double> pt1(2);
        pt1.mutable_at(0) = ox1[closest_idx];
        pt1.mutable_at(1) = oy1[closest_idx];
        last_obstacle[obstacle_mem_key] = py::make_tuple(pt0, pt1);
    }

    // Direction vector from seg_a to seg_b
    double dx = seg_b.x - seg_a.x;
    double dy = seg_b.y - seg_a.y;
    double dir_norm = std::hypot(dx, dy);

    Point subgoal_left_pt{px, py};
    Point subgoal_right_pt{px, py};
    bool left_valid = false;
    bool right_valid = false;

    if (dir_norm > 0.0) {
        // Left: subgoal_direction = 1 -> theta = 1.5 * pi -> cos = 0, sin = -1
        double ux_l = -dy / dir_norm;
        double uy_l = dx / dir_norm;
        auto [found_l, sl_x, sl_y, last_hit_l, rect_last_l] = find_subgoal(
            px, py, ux_l, uy_l, subgoal_distance, 1, clearance,
            ox0, oy0, ox1, oy1, n, has_rect, rmin_x, rmax_x, rmin_y, rmax_y
        );
        if (found_l) {
            subgoal_left_pt = {sl_x, sl_y};
            left_valid = true;
        } else {
            bool blocked = rect_last_l || (last_hit_l == closest_idx);
            if (!blocked) {
                subgoal_left_pt = {px, py};
                left_valid = true;
            }
        }

        // Right: subgoal_direction = 0 -> theta = 0.5 * pi -> cos = 0, sin = 1
        double ux_r = dy / dir_norm;
        double uy_r = -dx / dir_norm;
        auto [found_r, sr_x, sr_y, last_hit_r, rect_last_r] = find_subgoal(
            px, py, ux_r, uy_r, subgoal_distance, 1, clearance,
            ox0, oy0, ox1, oy1, n, has_rect, rmin_x, rmax_x, rmin_y, rmax_y
        );
        if (found_r) {
            subgoal_right_pt = {sr_x, sr_y};
            right_valid = true;
        } else {
            bool blocked = rect_last_r || (last_hit_r == closest_idx);
            if (!blocked) {
                subgoal_right_pt = {px, py};
                right_valid = true;
            }
        }
    }

    if (left_valid) {
        left_valid = (fb_min_x <= subgoal_left_pt.x && subgoal_left_pt.x <= fb_max_x &&
                      fb_min_y <= subgoal_left_pt.y && subgoal_left_pt.y <= fb_max_y);
    }
    if (right_valid) {
        right_valid = (fb_min_x <= subgoal_right_pt.x && subgoal_right_pt.x <= fb_max_x &&
                       fb_min_y <= subgoal_right_pt.y && subgoal_right_pt.y <= fb_max_y);
    }

    Point best_subgoal;
    int chosen_side = -1;

    py::object memory_key = py::none();
    std::optional<int> prev_side = std::nullopt;
    if (robot_id >= 0) {
        double r_px = round_to(px, precision);
        double r_py = round_to(py, precision);
        memory_key = py::make_tuple(robot_id, recursion_length, py::make_tuple(r_px, r_py));
        if (last_detour_side.contains(memory_key)) {
            prev_side = last_detour_side[memory_key].cast<int>();
        }
    }

    if (left_valid && right_valid) {
        double dist_left = std::hypot(subgoal_left_pt.x - target.x, subgoal_left_pt.y - target.y);
        double dist_right = std::hypot(subgoal_right_pt.x - target.x, subgoal_right_pt.y - target.y);

        double ghost_dx = ox1[closest_idx] - ox0[closest_idx];
        double ghost_dy = oy1[closest_idx] - oy0[closest_idx];
        double ghost_len = std::hypot(ghost_dx, ghost_dy);

        if (ghost_len > moving_ghost_min_m && (!is_static || !is_static[closest_idx])) {
            double along_left = ((subgoal_left_pt.x - ox0[closest_idx]) * ghost_dx +
                                 (subgoal_left_pt.y - oy0[closest_idx]) * ghost_dy) / ghost_len;
            double along_right = ((subgoal_right_pt.x - ox0[closest_idx]) * ghost_dx +
                                  (subgoal_right_pt.y - oy0[closest_idx]) * ghost_dy) / ghost_len;
            if (std::abs(along_left - along_right) > moving_ghost_min_m) {
                if (along_left < along_right) {
                    dist_left = 0.0;
                    dist_right = 1.0;
                } else {
                    dist_left = 1.0;
                    dist_right = 0.0;
                }
            }
        }

        if (prev_side.has_value() && prev_side.value() == 1 && dist_left <= dist_right + detour_switch_margin) {
            best_subgoal = subgoal_left_pt;
            chosen_side = 1;
        } else if (prev_side.has_value() && prev_side.value() == 0 && dist_right <= dist_left + detour_switch_margin) {
            best_subgoal = subgoal_right_pt;
            chosen_side = 0;
        } else if (dist_left < dist_right) {
            best_subgoal = subgoal_left_pt;
            chosen_side = 1;
        } else {
            best_subgoal = subgoal_right_pt;
            chosen_side = 0;
        }
    } else if (left_valid) {
        best_subgoal = subgoal_left_pt;
        chosen_side = 1;
    } else if (right_valid) {
        best_subgoal = subgoal_right_pt;
        chosen_side = 0;
    } else {
        out_trajectory.push_back({seg_a, seg_b});
        return;
    }

    if (robot_id >= 0 && !memory_key.is_none()) {
        last_detour_side[memory_key] = chosen_side;
    }

    check_segment_recursive(
        seg_a, best_subgoal, recursion_length + 1, max_recursion_length,
        target, fb_min_x, fb_max_x, fb_min_y, fb_max_y, robot_id,
        clearance, subgoal_distance, has_rect, rmin_x, rmax_x, rmin_y, rmax_y,
        detour_switch_margin, moving_ghost_min_m, precision,
        ox0, oy0, ox1, oy1, n, is_static, last_detour_side, last_obstacle,
        out_trajectory
    );
    check_segment_recursive(
        best_subgoal, seg_b, recursion_length + 1, max_recursion_length,
        target, fb_min_x, fb_max_x, fb_min_y, fb_max_y, robot_id,
        clearance, subgoal_distance, has_rect, rmin_x, rmax_x, rmin_y, rmax_y,
        detour_switch_margin, moving_ghost_min_m, precision,
        ox0, oy0, ox1, oy1, n, is_static, last_detour_side, last_obstacle,
        out_trajectory
    );
}

Point smooth_path_cpp(
    const std::vector<Segment>& trajectory,
    Point target, Point robot_pos,
    double projection_distance, double clearance,
    const double* ox0, const double* oy0, const double* ox1, const double* oy1, size_t n
) {
    size_t start_idx = 0;
    while (trajectory.size() - start_idx > 1 &&
           std::hypot(trajectory[start_idx].b.x - robot_pos.x, trajectory[start_idx].b.y - robot_pos.y) == 0.0) {
        start_idx++;
    }
    size_t remaining = trajectory.size() - start_idx;
    if (remaining <= 1) {
        return target;
    }

    Point next_pt = trajectory[start_idx].b;
    double dir_x = next_pt.x - robot_pos.x;
    double dir_y = next_pt.y - robot_pos.y;
    double norm = std::hypot(dir_x, dir_y);
    double unit_x = dir_x / norm;
    double unit_y = dir_y / norm;

    double safe_dist = clamp_to_obstacle_clearance(
        robot_pos.x, robot_pos.y, unit_x, unit_y, projection_distance, clearance,
        ox0, oy0, ox1, oy1, n
    );
    Point new_target{robot_pos.x + unit_x * safe_dist, robot_pos.y + unit_y * safe_dist};

    double dist_new_target = std::hypot(new_target.x - robot_pos.x, new_target.y - robot_pos.y);
    double dist_trajectory = std::hypot(robot_pos.x - next_pt.x, robot_pos.y - next_pt.y);

    if (dist_new_target < dist_trajectory) {
        return next_pt;
    } else {
        Point p1_0 = trajectory[start_idx + 1].a;
        Point p1_1 = trajectory[start_idx + 1].b;
        auto [cx, cy] = closest_point_on_segment_xy(new_target.x, new_target.y, p1_0.x, p1_0.y, p1_1.x, p1_1.y);
        return {(cx + new_target.x) / 2.0, (cy + new_target.y) / 2.0};
    }
}

// ============================================================================
// Pybind11 Module Wrapping
// ============================================================================

PYBIND11_MODULE(_fastpathplanner, m) {
    m.doc() = "FastPathPlanner C++ native accelerated geometry core";

    m.def("collides_nb", [](
        double ax, double ay, double bx, double by,
        double clearance,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox0,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy0,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox1,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy1,
        int sticky_idx, double switch_margin
    ) {
        size_t n = ox0.size();
        const double* d_ox0 = ox0.data();
        const double* d_oy0 = oy0.data();
        const double* d_ox1 = ox1.data();
        const double* d_oy1 = oy1.data();
        int idx;
        bool from_sticky;
        double px, py;
        {
            py::gil_scoped_release release;
            auto res = collides(
                ax, ay, bx, by, clearance,
                d_ox0, d_oy0, d_ox1, d_oy1, n,
                sticky_idx, switch_margin
            );
            idx = std::get<0>(res);
            from_sticky = std::get<1>(res);
            px = std::get<2>(res);
            py = std::get<3>(res);
        }
        return py::make_tuple(idx, from_sticky, px, py);
    });

    m.def("find_subgoal_nb", [](
        double opx, double opy, double ux, double uy,
        double subgoal_distance, int multiple, double clearance,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox0,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy0,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox1,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy1,
        bool has_rect, double rmin_x, double rmax_x, double rmin_y, double rmax_y
    ) {
        size_t n = ox0.size();
        const double* d_ox0 = ox0.data();
        const double* d_oy0 = oy0.data();
        const double* d_ox1 = ox1.data();
        const double* d_oy1 = oy1.data();
        bool found;
        double sx, sy;
        int last_hit;
        bool rect_last;
        {
            py::gil_scoped_release release;
            auto res = find_subgoal(
                opx, opy, ux, uy, subgoal_distance, multiple, clearance,
                d_ox0, d_oy0, d_ox1, d_oy1, n,
                has_rect, rmin_x, rmax_x, rmin_y, rmax_y
            );
            found = std::get<0>(res);
            sx = std::get<1>(res);
            sy = std::get<2>(res);
            last_hit = std::get<3>(res);
            rect_last = std::get<4>(res);
        }
        return py::make_tuple(found, sx, sy, last_hit, rect_last);
    });

    m.def("clamp_to_obstacle_clearance_nb", [](
        double ox, double oy, double ux, double uy, double max_distance, double clearance,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox0,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy0,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox1,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy1
    ) {
        size_t n = ox0.size();
        const double* d_ox0 = ox0.data();
        const double* d_oy0 = oy0.data();
        const double* d_ox1 = ox1.data();
        const double* d_oy1 = oy1.data();
        double res;
        {
            py::gil_scoped_release release;
            res = clamp_to_obstacle_clearance(
                ox, oy, ux, uy, max_distance, clearance,
                d_ox0, d_oy0, d_ox1, d_oy1, n
            );
        }
        return res;
    });

    m.def("sanitize_target_nb", [](
        double tx, double ty, double rx, double ry, double clearance,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox0,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy0,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox1,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy1,
        py::array_t<double, py::array::c_style | py::array::forcecast> exempt
    ) {
        size_t n = ox0.size();
        size_t n_exempt = exempt.size() / 4;
        const double* d_ox0 = ox0.data();
        const double* d_oy0 = oy0.data();
        const double* d_ox1 = ox1.data();
        const double* d_oy1 = oy1.data();
        const double* d_ex = exempt.data();
        double stx, sty;
        {
            py::gil_scoped_release release;
            auto res = sanitize_target(
                tx, ty, rx, ry, clearance,
                d_ox0, d_oy0, d_ox1, d_oy1, n,
                d_ex, n_exempt
            );
            stx = res.first;
            sty = res.second;
        }
        return py::make_tuple(stx, sty);
    });

    m.def("check_segment_cpp", [](
        py::sequence segment,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox0,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy0,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox1,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy1,
        std::optional<py::array_t<bool, py::array::c_style | py::array::forcecast>> is_static_opt,
        int recursion_length, int max_recursion_length,
        py::sequence target_seq,
        py::sequence bounds_seq,
        std::optional<int> robot_id_opt,
        double clearance, double subgoal_distance,
        std::optional<py::sequence> forbidden_rect_opt,
        double detour_switch_margin, double moving_ghost_min_m, int precision,
        py::dict last_detour_side,
        py::dict last_obstacle
    ) {
        py::sequence p_a = segment[0].cast<py::sequence>();
        py::sequence p_b = segment[1].cast<py::sequence>();
        Point seg_a{p_a[0].cast<double>(), p_a[1].cast<double>()};
        Point seg_b{p_b[0].cast<double>(), p_b[1].cast<double>()};
        Point target{target_seq[0].cast<double>(), target_seq[1].cast<double>()};

        double fb_min_x = bounds_seq[0].cast<double>();
        double fb_max_x = bounds_seq[1].cast<double>();
        double fb_min_y = bounds_seq[2].cast<double>();
        double fb_max_y = bounds_seq[3].cast<double>();

        int robot_id = robot_id_opt.value_or(-1);

        bool has_rect = forbidden_rect_opt.has_value() && !forbidden_rect_opt->is_none();
        double rmin_x = 0.0, rmax_x = 0.0, rmin_y = 0.0, rmax_y = 0.0;
        if (has_rect) {
            py::sequence r = forbidden_rect_opt.value();
            rmin_x = r[0].cast<double>();
            rmax_x = r[1].cast<double>();
            rmin_y = r[2].cast<double>();
            rmax_y = r[3].cast<double>();
        }

        size_t n = ox0.size();
        const bool* is_static_ptr = nullptr;
        if (is_static_opt.has_value()) {
            is_static_ptr = is_static_opt.value().data();
        }

        std::vector<Segment> out_trajectory;
        check_segment_recursive(
            seg_a, seg_b, recursion_length, max_recursion_length,
            target, fb_min_x, fb_max_x, fb_min_y, fb_max_y, robot_id,
            clearance, subgoal_distance, has_rect, rmin_x, rmax_x, rmin_y, rmax_y,
            detour_switch_margin, moving_ghost_min_m, precision,
            ox0.data(), oy0.data(), ox1.data(), oy1.data(), n,
            is_static_ptr, last_detour_side, last_obstacle,
            out_trajectory
        );

        py::list traj_list;
        double total_len = 0.0;
        for (const auto& s : out_trajectory) {
            py::array_t<double> a(2);
            a.mutable_at(0) = s.a.x;
            a.mutable_at(1) = s.a.y;
            py::array_t<double> b(2);
            b.mutable_at(0) = s.b.x;
            b.mutable_at(1) = s.b.y;
            traj_list.append(py::make_tuple(a, b));
            total_len += std::hypot(s.b.x - s.a.x, s.b.y - s.a.y);
        }
        return py::make_tuple(traj_list, total_len);
    });

    m.def("smooth_path_cpp", [](
        py::sequence trajectory_seq,
        py::sequence target_seq,
        py::sequence robot_pos_seq,
        double projection_distance, double clearance,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox0,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy0,
        py::array_t<double, py::array::c_style | py::array::forcecast> ox1,
        py::array_t<double, py::array::c_style | py::array::forcecast> oy1
    ) {
        std::vector<Segment> trajectory;
        for (auto item : trajectory_seq) {
            py::sequence s = item.cast<py::sequence>();
            py::sequence sa = s[0].cast<py::sequence>();
            py::sequence sb = s[1].cast<py::sequence>();
            trajectory.push_back({
                {sa[0].cast<double>(), sa[1].cast<double>()},
                {sb[0].cast<double>(), sb[1].cast<double>()}
            });
        }
        Point target{target_seq[0].cast<double>(), target_seq[1].cast<double>()};
        Point robot_pos{robot_pos_seq[0].cast<double>(), robot_pos_seq[1].cast<double>()};

        size_t n = ox0.size();
        Point res = smooth_path_cpp(
            trajectory, target, robot_pos, projection_distance, clearance,
            ox0.data(), oy0.data(), ox1.data(), oy1.data(), n
        );
        py::array_t<double> out(2);
        out.mutable_at(0) = res.x;
        out.mutable_at(1) = res.y;
        return out;
    });
}
