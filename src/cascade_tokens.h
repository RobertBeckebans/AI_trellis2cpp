/*
** cascade_tokens.h — HR-scaffold quantization and the cascade token budget.
**
** The cascade tiers quantize the shape decoder's 512^3 candidate coordinates
** onto a (resolution / 16)^3 scaffold and run the HR flow on the deduplicated
** set. TRELLIS.2 bounds that set rather than the resolution: while the set
** would reach max_num_tokens, the requested resolution steps down by 128 until
** it either fits or hits 1024 — the tier that is known to fit. So the HR flow
** at 1536 costs at most what it costs at 1024 today, and a 1536 request can
** never come out worse than a 1024 one.
**
** Header-only and free of ggml/model state, so the C-ABI pipeline
** (src/trellis2_capi.cpp) and tests/test_cascade_tokens.cpp share exactly one
** implementation of the formula. See docs/plan/1536-cascade.md (D2, D3).
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace t2cascade
{

// TRELLIS.2's default HR token budget, independent of the tier.
const int default_max_num_tokens = 49152;

// The floor the loop may fall back to, and the step it falls in.
const int min_resolution		 = 1024;
const int resolution_step		 = 128;

// The decoder's fixed scaffold-to-grid factor: resolution / 16 voxels per axis.
const int grid_divisor			 = 16;

// Pack a coordinate triple into one hash key. Coordinates stay well below 2^20
// at every tier (96 at 1536), so the three fields never collide.
inline uint64_t voxel_key( int32_t x, int32_t y, int32_t z )
{
	return ( ( uint64_t )( uint32_t )x << 40 ) | ( ( uint64_t )( uint32_t )y << 20 ) | ( uint64_t )( uint32_t )z;
}

// Quantize [x,y,z,...] candidates from an lr_res^3 grid onto hr_grid^3, keeping
// the first occurrence of each cell.
//
// The scale and the truncation are TRELLIS.2's, not a rounding choice of ours:
// upstream computes ((c + 0.5) / lr_resolution * (resolution // 16)).int().
// Pixal3D uses a different form (round(), and grid_res - 1 as the factor) —
// that belongs to the projection-conditioning plan, not here.
inline void quantize_scaffold( const std::vector<int32_t>& up_coords, int lr_res, int hr_grid, std::vector<int32_t>& out )
{
	out.clear();
	std::unordered_set<uint64_t> seen;
	seen.reserve( up_coords.size() / 3 );
	for( size_t i = 0; i + 2 < up_coords.size(); i += 3 ) {
		const int32_t qx = ( int32_t )( ( up_coords[i] + 0.5f ) / lr_res * hr_grid );
		const int32_t qy = ( int32_t )( ( up_coords[i + 1] + 0.5f ) / lr_res * hr_grid );
		const int32_t qz = ( int32_t )( ( up_coords[i + 2] + 0.5f ) / lr_res * hr_grid );
		if( seen.insert( voxel_key( qx, qy, qz ) ).second ) {
			out.push_back( qx );
			out.push_back( qy );
			out.push_back( qz );
		}
	}
}

// What the budget loop settled on.
struct selection {
	int resolution = 0; // achieved pipeline resolution (a multiple of 128)
	int grid	   = 0; // scaffold resolution, resolution / 16
	int tokens	   = 0; // voxels in the resulting scaffold
	int reductions = 0; // how often the loop had to step down (0 = requested)
};

// Quantize at requested_res and step down by 128 while the scaffold would reach
// max_num_tokens, stopping at min_resolution. `out` holds the scaffold of the
// resolution that is returned.
//
// At requested_res == 1024 this quantizes once and breaks on the floor, which
// is what the port did before the 1536 tier existed — the 1024 path is
// unchanged by construction, not by coincidence.
inline selection select_scaffold( const std::vector<int32_t>& up_coords, int lr_res, int requested_res, int max_num_tokens, std::vector<int32_t>& out )
{
	selection s;
	s.resolution = requested_res;
	for( ;; ) {
		s.grid = s.resolution / grid_divisor;
		quantize_scaffold( up_coords, lr_res, s.grid, out );
		s.tokens = ( int )( out.size() / 3 );
		// `<=` on the floor rather than upstream's `==`: identical for the two
		// shipped tiers, but it also terminates if a caller ever asks below it.
		if( s.tokens < max_num_tokens || s.resolution <= min_resolution )
			break;
		s.resolution -= resolution_step;
		s.reductions++;
	}
	return s;
}

} // namespace t2cascade
