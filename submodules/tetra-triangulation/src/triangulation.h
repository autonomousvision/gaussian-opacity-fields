#include <cuda_runtime.h>
#include <vector>
#include "utils/exception.h"

std::vector<uint4> triangulate(size_t num_points, float3 *points);
float find_average_spacing(size_t num_points, float3 *points);