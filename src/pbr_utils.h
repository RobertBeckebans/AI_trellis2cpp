#pragma once

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace t2pbr
{

inline uint64_t voxel_key( int32_t x, int32_t y, int32_t z )
{
	return ( ( uint64_t )( uint32_t )x << 42 ) | ( ( uint64_t )( uint32_t )y << 21 ) | ( uint64_t )( uint32_t )z;
}

// Trilinearly sample a sparse voxel field. query_vox is expressed in voxel
// coordinates, exactly like upstream's (xyz - origin) / voxel_size. Missing
// corners do not contribute and present weights are renormalized, matching the
// reference helper used for per-vertex texture validation.
inline void sample_sparse_trilinear( const float* feats, int n_voxels, int channels, const int32_t* coords, const float* query_vox, int n_queries, float* out, float* out_weights = nullptr )
{
	std::unordered_map<uint64_t, int> rows;
	rows.reserve( ( size_t )n_voxels * 2 );
	for( int i = 0; i < n_voxels; ++i ) {
		rows[voxel_key( coords[( size_t )i * 3], coords[( size_t )i * 3 + 1], coords[( size_t )i * 3 + 2] )] = i;
	}

	for( int q = 0; q < n_queries; ++q ) {
		float* dst = out + ( size_t )q * channels;
		for( int c = 0; c < channels; ++c )
			dst[c] = 0.0f;

		const float*  p	 = query_vox + ( size_t )q * 3;
		const int32_t bx = ( int32_t )std::floor( p[0] );
		const int32_t by = ( int32_t )std::floor( p[1] );
		const int32_t bz = ( int32_t )std::floor( p[2] );
		const float	  fx = p[0] - bx, fy = p[1] - by, fz = p[2] - bz;
		float		  wsum = 0.0f;

		for( int dx = 0; dx <= 1; ++dx )
			for( int dy = 0; dy <= 1; ++dy )
				for( int dz = 0; dz <= 1; ++dz ) {
					const float w = ( dx ? fx : 1.0f - fx ) * ( dy ? fy : 1.0f - fy ) * ( dz ? fz : 1.0f - fz );
					if( w == 0.0f ) continue;
					auto it = rows.find( voxel_key( bx + dx, by + dy, bz + dz ) );
					if( it == rows.end() ) continue;
					const float* src = feats + ( size_t )it->second * channels;
					for( int c = 0; c < channels; ++c )
						dst[c] += w * src[c];
					wsum += w;
				}
		if( wsum > 1e-6f ) {
			const float inv = 1.0f / wsum;
			for( int c = 0; c < channels; ++c )
				dst[c] *= inv;
		}
		if( out_weights ) out_weights[q] = wsum;
	}
}

// Detect the characteristic failed texture decode where nearly every vertex
// has every material channel pinned to one. Requiring all channels avoids
// rejecting legitimate white, fully opaque, metallic, or rough materials.
inline bool is_collapsed_saturated( const float* pbr, int n_vertices, int channels = 6, float threshold = 0.999f, int permille = 995 )
{
	if( !pbr || n_vertices <= 0 || channels <= 0 ) return false;
	size_t saturated = 0;
	for( int v = 0; v < n_vertices; ++v ) {
		bool all_one = true;
		for( int c = 0; c < channels; ++c ) {
			all_one = all_one && pbr[( size_t )v * channels + c] >= threshold;
		}
		saturated += all_one;
	}
	return saturated * 1000 >= ( size_t )n_vertices * permille;
}

} // namespace t2pbr
