// Chunked vs. unchunked finest decoder level must produce the same PBR.
//
// shape_dec_final_level_chunked splits the finest level along voxels so no
// mul_mat outgrows the backend's column limit (see tests/test_large_rows). The
// split is only legal because norm and silu are per-row and therefore commute
// with the neighbour gather; this test is what holds that claim honest.
//
// The texture decoder is used because it replays an explicit subdivision, so
// the voxel count at every level is chosen here rather than predicted from the
// latents -- 8 latent voxels fully subdivided four times give 32768 rows at the
// finest level, enough to force several blocks at a small TRELLIS2_CHUNK_ROWS
// while staying fast.
//
// Both the forced small block and the production block size are checked, the
// latter because an end-to-end A/B is useless here: the flow samplers are not
// reproducible run to run, so two full generations differ in voxel count for
// reasons that have nothing to do with the split. This is the deterministic
// instrument for that question.
//
// Usage: test_chunked_decode [tex_dec.gguf] [latent extent, default 2]
//        extent 5 gives 512000 fine voxels, the scale of a real 512^3 decode.
// Exit:  0 identical, 1 diverged, 77 model missing.

#include "trellis2.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

// An empty value clears the override, so the decoder falls back to the block
// size a real generation uses.
void set_chunk( const char* value )
{
#ifdef _WIN32
	_putenv_s( "TRELLIS2_CHUNK_ROWS", value );
#else
	if( value[0] )
		setenv( "TRELLIS2_CHUNK_ROWS", value, 1 );
	else
		unsetenv( "TRELLIS2_CHUNK_ROWS" );
#endif
}

float lcg( uint32_t& s )
{
	s = s * 1664525u + 1013904223u;
	return ( float )( ( int32_t )( s >> 8 ) % 2000 - 1000 ) * 0.002f;
}

// Every parent keeps all 8 children, so level L has 8x the voxels of L-1 and the
// coords stay a solid block -- neighbour lookups mostly hit, which is what the
// masked conv path we are testing actually does in a real decode.
trellis2_subdiv_level full_subdivision( const std::vector<int32_t>& coarse )
{
	trellis2_subdiv_level s;
	const size_t		  n = coarse.size() / 3;
	s.cidx.reserve( n * 8 );
	s.fine_coords.reserve( n * 24 );
	for( size_t v = 0; v < n; ++v ) {
		for( int o = 0; o < 8; ++o ) {
			s.cidx.push_back( ( int32_t )( o + 8 * v ) );
			s.fine_coords.push_back( 2 * coarse[v * 3] + ( o & 1 ) );
			s.fine_coords.push_back( 2 * coarse[v * 3 + 1] + ( ( o >> 1 ) & 1 ) );
			s.fine_coords.push_back( 2 * coarse[v * 3 + 2] + ( ( o >> 2 ) & 1 ) );
		}
	}
	return s;
}

} // namespace

int main( int argc, char** argv )
{
	const std::string path	 = argc > 1 ? argv[1] : "ggufs/tex_dec_f16.gguf";
	const int		  extent = argc > 2 ? std::atoi( argv[2] ) : 2;
	if( extent < 1 ) {
		std::fprintf( stderr, "bad extent\n" );
		return 1;
	}

	std::string				  err;
	trellis2_shape_dec_model* m = trellis2_tex_dec_load( path, true, &err );
	if( !m ) {
		std::printf( "tex_dec not loadable (%s): %s -- skipping\n", path.c_str(), err.c_str() );
		return 77;
	}
	const trellis2_shape_dec_hparams& hp = trellis2_shape_dec_hparams_of( m );

	// Solid latent block; four full subdivisions multiply it by 8^4 = 4096.
	std::vector<int32_t>			  coords;
	for( int x = 0; x < extent; ++x )
		for( int y = 0; y < extent; ++y )
			for( int z = 0; z < extent; ++z ) {
				coords.push_back( x );
				coords.push_back( y );
				coords.push_back( z );
			}
	const int						   L = ( int )( coords.size() / 3 );

	std::vector<trellis2_subdiv_level> subs;
	{
		std::vector<int32_t> cur = coords;
		for( int i = 0; i < hp.n_levels - 1; ++i ) {
			subs.push_back( full_subdivision( cur ) );
			cur = subs.back().fine_coords;
		}
	}
	const size_t	   fine = subs.back().fine_coords.size() / 3;

	uint32_t		   s = 20260810u;
	std::vector<float> slat( ( size_t )L * hp.latent_channels );
	for( auto& v : slat )
		v = lcg( s );

	std::printf( "model: %s\n", path.c_str() );
	std::printf( "latent voxels %d -> %zu fine voxels over %d levels\n\n", L, fine, hp.n_levels );

	std::vector<float>	 ref;
	std::vector<int32_t> rc;

	set_chunk( "100000000" ); // far above `fine`: single graph, the old path
	if( !trellis2_tex_dec_decode( m, slat.data(), L, coords.data(), subs, ref, rc, &err ) ) {
		std::printf( "unchunked decode failed: %s\n", err.c_str() );
		trellis2_shape_dec_free( m );
		return 1;
	}

	// "" restores the production block size, which is the configuration a real
	// generation runs; "997" is not a divisor, so the last block is ragged.
	const char* cases[]	 = { "", "997" };
	int			failures = 0;
	for( const char* c : cases ) {
		std::vector<float>	 got;
		std::vector<int32_t> gc;
		set_chunk( c );
		if( !trellis2_tex_dec_decode( m, slat.data(), L, coords.data(), subs, got, gc, &err ) ) {
			std::printf( "  block=%-9s decode failed: %s\n", c[0] ? c : "default", err.c_str() );
			++failures;
			continue;
		}
		if( ref.size() != got.size() || rc != gc ) {
			std::printf( "  block=%-9s SHAPE MISMATCH feats %zu vs %zu -> FAIL\n", c[0] ? c : "default", ref.size(), got.size() );
			++failures;
			continue;
		}
		// Both runs do the same float ops per row; only the graph partitioning
		// differs, so any drift would come from the backend picking a different
		// GEMM tiling for a narrower matrix.
		double worst = 0.0;
		size_t at = 0, nbad = 0;
		for( size_t i = 0; i < ref.size(); ++i ) {
			const double d = std::fabs( ( double )got[i] - ( double )ref[i] );
			if( d > worst ) {
				worst = d;
				at	  = i;
			}
			if( d > 1e-5 )
				++nbad;
		}
		std::printf( "  block=%-9s %zu values, %zu over 1e-5, max|d| = %.3g "
					 "(voxel %zu channel %zu) -> %s\n",
			c[0] ? c : "default",
			ref.size(),
			nbad,
			worst,
			at / hp.out_channels,
			at % hp.out_channels,
			nbad ? "FAIL" : "OK" );
		if( nbad )
			++failures;
	}
	trellis2_shape_dec_free( m );
	return failures ? 1 : 0;
}
