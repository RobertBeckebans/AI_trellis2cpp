// Helpers for comparing C++ activations against PyTorch reference dumps
// (dumps/reference_*.gguf written by scripts/dump_*_reference.py).
//
// Same approach as depth-anything.cpp's tests/parity.hpp: the dump format is
// GGUF itself, tensors are flat f32 in the reference's row-major order, and
// the gate is elementwise |got - ref| <= atol + rtol * |ref|.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace t2_parity
{

struct baseline {
	gguf_context* gguf = nullptr;
	ggml_context* ctx  = nullptr;

	bool		  open( const std::string& path )
	{
		gguf_init_params p;
		p.no_alloc = false; // load tensor data into ctx
		p.ctx	   = &ctx;
		gguf	   = gguf_init_from_file( path.c_str(), p );
		return gguf != nullptr;
	}

	bool load( const std::string& name, std::vector<float>& out ) const
	{
		ggml_tensor* t = ggml_get_tensor( ctx, name.c_str() );
		if( !t || t->type != GGML_TYPE_F32 ) return false;
		out.resize( ( size_t )ggml_nelements( t ) );
		std::memcpy( out.data(), t->data, out.size() * sizeof( float ) );
		return true;
	}

	bool has( const std::string& name ) const { return ggml_get_tensor( ctx, name.c_str() ) != nullptr; }

	~baseline()
	{
		if( gguf ) gguf_free( gguf );
		if( ctx ) ggml_free( ctx );
	}
};

struct compare_stats {
	double max_abs	= 0.0;
	double mean_abs = 0.0;
	double rel_l2	= 0.0;
	size_t worst	= 0;
	bool   ok		= false;
};

// Elementwise gate |got - ref| <= atol + rtol*|ref|, with summary line.
inline bool compare( const std::vector<float>& got, const std::vector<float>& ref, const std::string& label, double atol, double rtol, compare_stats* stats_out = nullptr )
{
	compare_stats st;
	if( got.size() != ref.size() || got.empty() ) {
		std::printf( "[%-18s] SIZE MISMATCH got=%zu ref=%zu -> FAIL\n", label.c_str(), got.size(), ref.size() );
		if( stats_out ) *stats_out = st;
		return false;
	}
	double sum_abs = 0.0, num = 0.0, den = 0.0;
	size_t nbad = 0;
	for( size_t i = 0; i < got.size(); ++i ) {
		const double d = std::fabs( ( double )got[i] - ( double )ref[i] );
		const double r = ( double )ref[i];
		num += d * d;
		den += r * r;
		sum_abs += d;
		if( d > st.max_abs ) {
			st.max_abs = d;
			st.worst   = i;
		}
		if( d > atol + rtol * std::fabs( r ) ) ++nbad;
	}
	st.mean_abs = sum_abs / ( double )got.size();
	st.rel_l2	= den > 0 ? std::sqrt( num / den ) : std::sqrt( num );
	st.ok		= nbad == 0;
	std::printf( "[%-18s] n=%-8zu max|d|=%-11.4g mean|d|=%-11.4g relL2=%-11.4g "
				 "(worst@%zu got=%.6g ref=%.6g) -> %s\n",
		label.c_str(),
		got.size(),
		st.max_abs,
		st.mean_abs,
		st.rel_l2,
		st.worst,
		( double )got[st.worst],
		( double )ref[st.worst],
		st.ok ? "OK" : "FAIL" );
	if( stats_out ) *stats_out = st;
	return st.ok;
}

} // namespace t2_parity
