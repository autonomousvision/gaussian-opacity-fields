#include "triangulation.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/compute_average_spacing.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <limits>

#include "utils/exception.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Triangulation;

typedef K::FT FT;
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

typedef Triangulation::Cell_handle Cell_handle;
typedef Triangulation::Vertex_handle Vertex_handle;
typedef Triangulation::Locate_type Locate_type;
typedef Triangulation::Point Point;

std::vector<uint4> triangulate(size_t num_points, float3* points) {
    std::vector<std::pair<Point, unsigned int>> L(num_points);
    for (size_t i = 0; i < num_points; i++) {
        const auto p = Point(
            points[i].x,
            points[i].y,
            points[i].z);
        L[i] = std::make_pair(p, i);
    }

    Triangulation T(L.begin(), L.end());
    if (!T.is_valid()) {
        throw Exception("Triangulation failed");
    }

    // Export
    std::vector<uint4> cells(T.number_of_finite_cells());
    unsigned int* cells_uint = reinterpret_cast<unsigned int*>(cells.data());

    size_t i = 0;
    for (auto cell : T.finite_cell_handles()) {
        for (int j = 0; j < 4; ++j) {
            cells_uint[i * 4 + j] = cell->vertex(j)->info();
        }
        i++;
    }

    // Fix locality
    // TODO!!
    // const unsigned int desired_cluster_size = 128;
    // int max_depth = std::ceil(std::log2(L.size() / desired_cluster_size));
    // std::shared_ptr<OctreeNode> octree = build_octree(L,
    //                                                   std::numeric_limits<float>.min(),
    //                                                   std::numeric_limits<float>.max(),
    //                                                   std::numeric_limits<float>.min(),
    //                                                   std::numeric_limits<float>.max(),
    //                                                   std::numeric_limits<float>.min(),
    //                                                   std::numeric_limits<float>.max(),
    //                                                   0, max_depth);
    
    return cells;
}

struct OctreeNode {
    std::array<std::shared_ptr<OctreeNode>, 8> children;
    std::vector<std::pair<Point, unsigned int>> points;
    // Other fields for octree structure
};

std::shared_ptr<OctreeNode> build_octree(const std::vector<std::pair<Point, unsigned int>>& points, float min_x, float max_x,
                         float min_y, float max_y, float min_z, float max_z, int depth, int max_depth) {
    if (depth == max_depth) {
        std::shared_ptr<OctreeNode> node;
        node->points = points;
        // Code to set other fields for the octree structure
        return node;
    }

    float mid_x = (min_x + max_x) / 2.0f;
    float mid_y = (min_y + max_y) / 2.0f;
    float mid_z = (min_z + max_z) / 2.0f;

    std::array<std::vector<std::pair<Point, unsigned int>>, 8> child_points;
    for (const auto& p : points) {
        int index = ((p.first.x() >= mid_x) << 2) | ((p.first.y() >= mid_y) << 1) | (p.first.z() >= mid_z);
        child_points[index].push_back(p);
    }

    std::shared_ptr<OctreeNode> node;
    for (int i = 0; i < 8; i++) {
        if (!child_points[i].empty()) {
            float child_min_x = i & 4 ? mid_x : min_x;
            float child_max_x = i & 4 ? max_x : mid_x;
            float child_min_y = i & 2 ? mid_y : min_y;
            float child_max_y = i & 2 ? max_y : mid_y;
            float child_min_z = i & 1 ? mid_z : min_z;
            float child_max_z = i & 1 ? max_z : mid_z;
            node->children[i] = build_octree(child_points[i], child_min_x, child_max_x,
                                             child_min_y, child_max_y, child_min_z, child_max_z,
                                             depth + 1, max_depth);
        }
    }
    return node;
}



float find_average_spacing(size_t num_points, float3* points) {
    const unsigned int nb_neighbors = 6;  // 1 ring
    std::vector<Point> L(num_points);
    for (size_t i = 0; i < num_points; i++) {
        const auto p = Point(
            points[i].x,
            points[i].y,
            points[i].z);
        L[i] = p;
    }

    FT average_spacing = CGAL::compute_average_spacing<Concurrency_tag>(L, nb_neighbors);
    return average_spacing;
}