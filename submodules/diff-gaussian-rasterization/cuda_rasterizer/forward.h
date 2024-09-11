/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* view2gaussian_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, const int H,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float tan_fovx, const float tan_fovy,
		const float kernel_size,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* view2gaussians,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const int W, const int H,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float2* subpixel_offset,
		const float2* points_xy_image,
		const float* features,
		const float* view2gaussian,
		const float* cov3Ds,
		const float* viewmatrix,
		const float3* means3D,
		const float3* scales,
		const float* depths,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		float* center_depth,
		float4* center_alphas,
		const float* bg_color,
		float* out_color);

	// Perform initial steps for each Point prior to integration.
	void preprocess_points(int PN, int D, int M,
		const float* points3D,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, const int H,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float tan_fovx, const float tan_fovy,
		float2* points2D,
		float* depths,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);
	
	// Main rasterization method.
	void integrate(
		const dim3 grid, dim3 block,
		const uint2* gaussian_ranges,
		const uint2* point_ranges,
		const uint32_t* gaussian_list,
		const uint32_t* point_list,
		const int W, const int H,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float2* subpixel_offset,
		const float2* points2D,
		const float* features,
		const float* view2gaussian,
		const float* cov3Ds,
		const float* viewmatrix,
		const float3* means3D,
		const float3* scales,
		const float* depths,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		// float* center_depth,
		// float4* center_alphas,
		const float* bg_color,
		float* out_color,
		float* out_alpha_integrated,
		float* out_color_integrated);
}


#endif