#include "pbr_utils.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool close( float a, float b, float eps = 1e-6f )
{
	return std::fabs( a - b ) <= eps;
}

int main()
{
	// Eight corners of a unit cube, with f(x,y,z) = x + 2y + 4z.
	int32_t coords[8 * 3];
	float	feats[8];
	int		n = 0;
	for( int x = 0; x <= 1; ++x )
		for( int y = 0; y <= 1; ++y )
			for( int z = 0; z <= 1; ++z ) {
				coords[3 * n]	  = x;
				coords[3 * n + 1] = y;
				coords[3 * n + 2] = z;
				feats[n++]		  = ( float )( x + 2 * y + 4 * z );
			}
	const float query[] = { 0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f };
	float		out[3], weights[3];
	t2pbr::sample_sparse_trilinear( feats, 8, 1, coords, query, 3, out, weights );
	if( !close( out[0], 0.0f ) || !close( out[1], 3.5f ) || !close( out[2], 7.0f ) || !close( weights[0], 1.0f ) || !close( weights[1], 1.0f ) || !close( weights[2], 1.0f ) ) {
		std::fprintf( stderr, "dense trilinear mismatch: %g %g %g\n", out[0], out[1], out[2] );
		return 1;
	}

	// Sparse boundary: only one weighted corner exists, so it is renormalized
	// instead of darkening toward absent voxels.
	const int32_t one_coord[] = { 0, 0, 0 };
	const float	  one_feat[]  = { 0.25f, 0.75f };
	const float	  half[]	  = { 0.5f, 0.5f, 0.5f };
	float		  sparse[2], sparse_weight;
	t2pbr::sample_sparse_trilinear( one_feat, 1, 2, one_coord, half, 1, sparse, &sparse_weight );
	if( !close( sparse[0], 0.25f ) || !close( sparse[1], 0.75f ) || !close( sparse_weight, 0.125f ) ) {
		std::fprintf( stderr, "sparse trilinear mismatch: %g %g w=%g\n", sparse[0], sparse[1], sparse_weight );
		return 1;
	}

	// Regression: the failed integrated texture path returned essentially an
	// all-one six-channel material. It must be rejected, without mistaking a
	// legitimate white base colour for a collapsed full PBR result.
	std::vector<float> collapsed( 1000 * 6, 1.0f );
	for( int v = 0; v < 5; ++v )
		collapsed[( size_t )v * 6] = 0.5f;
	if( !t2pbr::is_collapsed_saturated( collapsed.data(), 1000 ) ) {
		std::fprintf( stderr, "collapsed saturated material was not detected\n" );
		return 1;
	}
	collapsed[5 * 6] = 0.5f; // only 994/1000 now have all channels saturated
	if( t2pbr::is_collapsed_saturated( collapsed.data(), 1000 ) ) {
		std::fprintf( stderr, "saturation threshold is too aggressive\n" );
		return 1;
	}
	std::vector<float> white( 16 * 6, 1.0f );
	for( int v = 0; v < 16; ++v ) {
		white[( size_t )v * 6 + 3] = 0.0f;
		white[( size_t )v * 6 + 4] = 0.5f;
	}
	if( t2pbr::is_collapsed_saturated( white.data(), 16 ) ) {
		std::fprintf( stderr, "legitimate white material was rejected\n" );
		return 1;
	}

	std::puts( "RESULT: PASS" );
	return 0;
}
