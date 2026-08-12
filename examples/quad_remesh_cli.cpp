// Phase 1 smoke tool for the vendored AutoRemesher core: OBJ in, quad OBJ out.
//
// This deliberately talks to the vendored headers directly instead of going
// through a trellis2 API — the point is to prove the core builds, runs, and
// produces sane numbers *before* any library wiring exists (that is Phase 3's
// quad_remesh.{h,cpp}). See docs/plan/autoremesher-quad-remesh.md.
//
//   quad_remesh_cli in.obj out.obj [--target-quads N] [--edge-scaling f]
//                                  [--adaptivity f] [--anisotropy f]
//                                  [--sharp-edge deg] [--smooth-normal deg]
//                                  [--hard-surface]
//
// Reports the measurements Phase 1 asks for: wall time, peak RSS, quad ratio,
// boundary edges, and surface-area retention. The last one is the proxy for
// silently dropped islands — AutoRemesher skips islands whose parameterization
// throws, and the API does not report that, so a large area loss is the only
// signal the caller gets.

#include "quad_remesh.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#if defined( _WIN32 )
	#include <windows.h>
	#include <psapi.h>
#endif

namespace
{

// Peak resident set size in MiB, or -1 where we cannot ask cheaply.
double peak_rss_mib()
{
#if defined( _WIN32 )
	PROCESS_MEMORY_COUNTERS counters;
	if( GetProcessMemoryInfo( GetCurrentProcess(), &counters, sizeof( counters ) ) )
		return ( double )counters.PeakWorkingSetSize / ( 1024.0 * 1024.0 );
	return -1.0;
#elif defined( __linux__ )
	std::FILE* status = std::fopen( "/proc/self/status", "r" );
	if( !status )
		return -1.0;
	char   line[256];
	double kib = -1.0;
	while( std::fgets( line, sizeof( line ), status ) ) {
		if( std::strncmp( line, "VmHWM:", 6 ) == 0 ) {
			kib = std::atof( line + 6 );
			break;
		}
	}
	std::fclose( status );
	return kib < 0.0 ? -1.0 : kib / 1024.0;
#else
	return -1.0;
#endif
}

bool read_obj( const char* path, std::vector<float>& verts, std::vector<int32_t>& tris, std::string& err )
{
	std::FILE* file = std::fopen( path, "rb" );
	if( !file ) {
		err = std::string( "cannot open " ) + path;
		return false;
	}

	char line[1024];
	while( std::fgets( line, sizeof( line ), file ) ) {
		if( line[0] == 'v' && ( line[1] == ' ' || line[1] == '\t' ) ) {
			double x = 0.0, y = 0.0, z = 0.0;
			if( std::sscanf( line + 1, "%lf %lf %lf", &x, &y, &z ) == 3 )
				verts.push_back( ( float )x );
			verts.push_back( ( float )y );
			verts.push_back( ( float )z );
		} else if( line[0] == 'f' && ( line[1] == ' ' || line[1] == '\t' ) ) {
			// Collect the polygon's vertex indices, ignoring the /vt/vn parts.
			std::vector<size_t> face;
			const char*			cursor = line + 1;
			while( *cursor ) {
				while( *cursor == ' ' || *cursor == '\t' )
					++cursor;
				if( *cursor == '\0' || *cursor == '\r' || *cursor == '\n' )
					break;
				const long index = std::strtol( cursor, nullptr, 10 );
				if( index > 0 ) {
					face.push_back( ( size_t )( index - 1 ) );
				} else if( index < 0 ) {
					const long resolved = ( long )verts.size() + index;
					if( resolved < 0 ) {
						err = "negative OBJ index out of range";
						std::fclose( file );
						return false;
					}
					face.push_back( ( size_t )resolved );
				}
				while( *cursor && *cursor != ' ' && *cursor != '\t' )
					++cursor;
			}
			// Fan-triangulate; the remesher takes triangles only.
			for( size_t i = 2; i < face.size(); ++i ) {
				tris.push_back( ( int32_t )face[0] );
				tris.push_back( ( int32_t )face[i - 1] );
				tris.push_back( ( int32_t )face[i] );
			}
		}
	}
	std::fclose( file );

	for( const int32_t index : tris ) {
		if( index < 0 || ( size_t )index * 3 + 2 >= verts.size() ) {
			err = "OBJ face index out of range";
			return false;
		}
	}
	if( verts.empty() || tris.empty() ) {
		err = "OBJ has no triangles";
		return false;
	}
	return true;
}

// The pipeline's own wire format, so a real generation can be fed in directly
// instead of being routed through OBJ:
//   magic[8] u32 nv u32 nt f32[3nv] verts f32[3nv] normals
//   [T2MESH02: f32[5nv]] [T2MESH03: f32[6nv]] i32[3nt] tris
bool read_t2mesh( const char* path, std::vector<float>& verts, std::vector<int32_t>& tris, std::string& err )
{
	std::FILE* file = std::fopen( path, "rb" );
	if( !file ) {
		err = std::string( "cannot open " ) + path;
		return false;
	}
	char	 magic[9] = { 0 };
	uint32_t nv = 0, nt = 0;
	if( std::fread( magic, 1, 8, file ) != 8 || std::fread( &nv, 4, 1, file ) != 1 || std::fread( &nt, 4, 1, file ) != 1 ) {
		err = "bad T2MESH header";
		std::fclose( file );
		return false;
	}
	const bool legacy	= 0 == std::memcmp( magic, "T2MESH02", 8 );
	const bool textured = legacy || 0 == std::memcmp( magic, "T2MESH03", 8 );
	if( !textured && 0 != std::memcmp( magic, "T2MESH01", 8 ) ) {
		err = "unknown T2MESH magic";
		std::fclose( file );
		return false;
	}
	std::vector<float> raw( ( size_t )nv * 3 );
	bool			   ok = std::fread( raw.data(), sizeof( float ), raw.size(), file ) == raw.size();
	std::vector<float> skip( ( size_t )nv * 3 );
	ok = ok && std::fread( skip.data(), sizeof( float ), skip.size(), file ) == skip.size(); // normals
	if( ok && textured ) {
		std::vector<float> pbr( ( size_t )nv * ( legacy ? 5 : 6 ) );
		ok = std::fread( pbr.data(), sizeof( float ), pbr.size(), file ) == pbr.size();
	}
	std::vector<int32_t> idx( ( size_t )nt * 3 );
	ok = ok && std::fread( idx.data(), sizeof( int32_t ), idx.size(), file ) == idx.size();
	std::fclose( file );
	if( !ok ) {
		err = "truncated T2MESH";
		return false;
	}
	verts.reserve( ( size_t )nv * 3 );
	for( uint32_t i = 0; i < nv; ++i )
		verts.insert( verts.end(), raw.begin() + ( size_t )i * 3, raw.begin() + ( size_t )i * 3 + 3 );
	tris.reserve( ( size_t )nt * 3 );
	for( uint32_t t = 0; t < nt; ++t ) {
		const int32_t a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
		if( a < 0 || b < 0 || c < 0 || ( uint32_t )a >= nv || ( uint32_t )b >= nv || ( uint32_t )c >= nv ) {
			err = "T2MESH index out of range";
			return false;
		}
		tris.push_back( a );
		tris.push_back( b );
		tris.push_back( c );
	}
	return true;
}

bool write_obj( const char* path, const std::vector<float>& verts, const std::vector<int32_t>& faces, const std::vector<int32_t>& face_sizes, std::string& err )
{
	std::FILE* file = std::fopen( path, "wb" );
	if( !file ) {
		err = std::string( "cannot write " ) + path;
		return false;
	}
	std::fprintf( file, "# generated by trellis2.cpp quad_remesh_cli\n" );
	for( size_t i = 0; i + 2 < verts.size(); i += 3 )
		std::fprintf( file, "v %.8f %.8f %.8f\n", verts[i], verts[i + 1], verts[i + 2] );
	size_t offset = 0;
	for( const int32_t n : face_sizes ) {
		std::fprintf( file, "f" );
		for( int32_t k = 0; k < n; ++k )
			std::fprintf( file, " %d", faces[offset + ( size_t )k] + 1 );
		std::fprintf( file, "\n" );
		offset += ( size_t )n;
	}
	std::fclose( file );
	return true;
}

// Edges used by exactly one face. Zero means closed; anything else means the
// output has holes, which is expected and is precisely why this path does not
// replace CGAL Alpha Wrap for printing.
double triangle_area( const float* a, const float* b, const float* c )
{
	const double ab[3] = { ( double )b[0] - a[0], ( double )b[1] - a[1], ( double )b[2] - a[2] };
	const double ac[3] = { ( double )c[0] - a[0], ( double )c[1] - a[1], ( double )c[2] - a[2] };
	const double n[3]  = { ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0] };
	return 0.5 * std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
}

size_t boundary_edges( const std::vector<std::vector<size_t>>& faces )
{
	std::map<std::pair<size_t, size_t>, int> counts;
	for( const auto& face : faces ) {
		for( size_t i = 0; i < face.size(); ++i ) {
			const size_t a = face[i], b = face[( i + 1 ) % face.size()];
			counts[std::make_pair( a < b ? a : b, a < b ? b : a )] += 1;
		}
	}
	size_t open = 0;
	for( const auto& entry : counts ) {
		if( entry.second == 1 )
			++open;
	}
	return open;
}

void report_progress( void* tag, float progress, const char* status )
{
	( void )tag;
	std::fprintf( stderr, "\r  %3d%%  %-40s", ( int )( progress * 100.0f ), status ? status : "" );
	std::fflush( stderr );
}

} // namespace

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr,
			"usage: %s in.obj out.obj [--target-quads N] [--edge-scaling f]\n"
			"          [--adaptivity f] [--anisotropy f] [--sharp-edge deg]\n"
			"          [--smooth-normal deg] [--hard-surface]\n",
			argv[0] );
		return 2;
	}

	// Defaults mirror the upstream CLI (src/main.cpp).
	int	   target_quads		  = 50000;
	double edge_scaling		  = 0.0; // 0 -> leave unset, as upstream does
	double adaptivity		  = 1.0;
	double anisotropy		  = 1.0;
	double sharp_edge_deg	  = 90.0;
	double smooth_normal_deg  = 0.0;
	bool   hard_surface		  = false;
	bool   split_non_manifold = true;
	int	   input_budget		  = 0;
	double min_area			  = 0.5;

	for( int i = 3; i < argc; ++i ) {
		const bool has_value = i + 1 < argc;
		if( std::strcmp( argv[i], "--target-quads" ) == 0 && has_value )
			target_quads = std::atoi( argv[++i] );
		else if( std::strcmp( argv[i], "--edge-scaling" ) == 0 && has_value )
			edge_scaling = std::atof( argv[++i] );
		else if( std::strcmp( argv[i], "--adaptivity" ) == 0 && has_value )
			adaptivity = std::atof( argv[++i] );
		else if( std::strcmp( argv[i], "--anisotropy" ) == 0 && has_value )
			anisotropy = std::atof( argv[++i] );
		else if( std::strcmp( argv[i], "--sharp-edge" ) == 0 && has_value )
			sharp_edge_deg = std::atof( argv[++i] );
		else if( std::strcmp( argv[i], "--smooth-normal" ) == 0 && has_value )
			smooth_normal_deg = std::atof( argv[++i] );
		else if( std::strcmp( argv[i], "--hard-surface" ) == 0 )
			hard_surface = true;
		else if( std::strcmp( argv[i], "--no-split" ) == 0 )
			split_non_manifold = false;
		else if( std::strcmp( argv[i], "--input-budget" ) == 0 && has_value )
			input_budget = std::atoi( argv[++i] );
		else if( std::strcmp( argv[i], "--min-area" ) == 0 && has_value )
			min_area = std::atof( argv[++i] );
		else {
			std::fprintf( stderr, "unknown or incomplete option: %s\n", argv[i] );
			return 2;
		}
	}
	if( target_quads <= 0 ) {
		std::fprintf( stderr, "--target-quads must be positive\n" );
		return 2;
	}

	std::vector<float>	 in_verts;
	std::vector<int32_t> in_tris;
	std::string			 err;
	const size_t		 path_len  = std::strlen( argv[1] );
	const bool			 is_t2mesh = path_len > 7 && 0 == std::strcmp( argv[1] + path_len - 7, ".t2mesh" );
	if( !( is_t2mesh ? read_t2mesh( argv[1], in_verts, in_tris, err ) : read_obj( argv[1], in_verts, in_tris, err ) ) ) {
		std::fprintf( stderr, "%s\n", err.c_str() );
		return 1;
	}
	double in_area = 0.0;
	for( size_t t = 0; t + 2 < in_tris.size(); t += 3 )
		in_area += triangle_area( in_verts.data() + ( size_t )in_tris[t] * 3, in_verts.data() + ( size_t )in_tris[t + 1] * 3, in_verts.data() + ( size_t )in_tris[t + 2] * 3 );
	std::fprintf( stderr, "in : %zu verts  %zu tris  area %.6f\n", in_verts.size() / 3, in_tris.size() / 3, in_area );

	t2quad::QuadRemeshOptions opt;
	opt.target_quads		  = target_quads;
	opt.edge_scaling		  = ( float )edge_scaling;
	opt.adaptivity			  = ( float )adaptivity;
	opt.anisotropy			  = ( float )anisotropy;
	opt.sharp_edge_deg		  = ( float )sharp_edge_deg;
	opt.smooth_normal_deg	  = ( float )smooth_normal_deg;
	opt.hard_surface		  = hard_surface;
	opt.split_non_manifold	  = split_non_manifold;
	opt.input_triangle_budget = input_budget;
	opt.min_area_retained	  = ( float )min_area;

	std::vector<float>		out_verts;
	std::vector<int32_t>	out_faces, out_face_sizes;
	t2quad::QuadRemeshStats stats;
	const auto				started	 = std::chrono::steady_clock::now();
	const bool				ok		 = t2quad::remesh( in_verts, in_tris, opt, out_verts, out_faces, out_face_sizes, stats, err, report_progress, nullptr );
	const auto				finished = std::chrono::steady_clock::now();
	std::fprintf( stderr, "\n%-52s\n", "" );
	if( !ok ) {
		std::fprintf( stderr, "remesh failed: %s\n", err.c_str() );
		return 1;
	}

	const double seconds = std::chrono::duration<double>( finished - started ).count();
	const double rss	 = peak_rss_mib();
	std::fprintf( stderr, "prep: %d tris after cleanup  %d verts split  %d faces dropped\n", stats.input_tris_after_prep, stats.vertices_split, stats.faces_dropped );
	std::fprintf( stderr, "out: %zu verts  %zu faces\n", out_verts.size() / 3, out_face_sizes.size() );
	std::fprintf( stderr, "     quads %d  tris %d  ngons %d\n", stats.quads, stats.triangles, stats.ngons );
	std::fprintf( stderr, "     boundary edges %d%s\n", stats.boundary_edges, stats.boundary_edges ? "  (open surface)" : "  (closed)" );
	std::fprintf( stderr, "     area retained %.1f%%\n", 100.0 * stats.area_retained );
	std::fprintf( stderr, "     wall %.2f s", seconds );
	if( rss >= 0.0 )
		std::fprintf( stderr, "   peak RSS %.1f MiB", rss );
	std::fprintf( stderr, "\n" );

	if( !write_obj( argv[2], out_verts, out_faces, out_face_sizes, err ) ) {
		std::fprintf( stderr, "%s\n", err.c_str() );
		return 1;
	}
	std::fprintf( stderr, "wrote %s\n", argv[2] );
	return 0;
}
