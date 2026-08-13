#include "mesh_export.h"
#include "print_remesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct EdgeUse {
	int count	  = 0;
	int direction = 0;
};

namespace
{

// Deterministic bit-mixer, so the query set is identical on every platform and
// run. Nothing here may depend on std::rand.
uint32_t mix( uint32_t x )
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

float unit( uint32_t x )
{
	return ( float )( mix( x ) >> 8 ) / ( float )( 1u << 24 );
}

// A displaced grid: enough triangles for the BVH to actually branch, and a
// continuous per-vertex PBR field. Continuity matters for the backend
// comparison — where two triangles share an edge, either may legitimately win
// the nearest-point tie, and only a continuous field makes the two answers
// comparable instead of arbitrarily different.
void make_grid( int n, std::vector<float>& verts, std::vector<int32_t>& tris, std::vector<float>& pbr )
{
	verts.clear();
	tris.clear();
	pbr.clear();
	for( int j = 0; j <= n; ++j ) {
		for( int i = 0; i <= n; ++i ) {
			const float x = ( float )i / ( float )n * 2.0f - 1.0f;
			const float z = ( float )j / ( float )n * 2.0f - 1.0f;
			const float y = 0.35f * std::sin( 3.0f * x ) * std::cos( 2.5f * z );
			verts.push_back( x );
			verts.push_back( y );
			verts.push_back( z );
			// Six channels straight from position: continuous by construction.
			pbr.push_back( 0.5f * x + 0.5f );
			pbr.push_back( 0.5f * y + 0.5f );
			pbr.push_back( 0.5f * z + 0.5f );
			pbr.push_back( 0.25f * x + 0.5f );
			pbr.push_back( 0.25f * z + 0.5f );
			pbr.push_back( 1.0f );
		}
	}
	for( int j = 0; j < n; ++j ) {
		for( int i = 0; i < n; ++i ) {
			const int32_t a = j * ( n + 1 ) + i;
			const int32_t b = a + 1;
			const int32_t c = a + ( n + 1 );
			const int32_t d = c + 1;
			tris.push_back( a );
			tris.push_back( c );
			tris.push_back( d );
			tris.push_back( a );
			tris.push_back( d );
			tris.push_back( b );
		}
	}
}

// A UV sphere, whose exact normal at every vertex is the unit position. That
// makes the projected shading normal checkable against a closed form instead of
// against another approximation. The poles emit degenerate triangles on purpose
// — prepare_faces has to filter them, exactly as it does for the dense meshes.
void make_sphere( int nu, int nv, std::vector<float>& verts, std::vector<int32_t>& tris, std::vector<float>& pbr, std::vector<float>& normals )
{
	verts.clear();
	tris.clear();
	pbr.clear();
	normals.clear();
	const double pi = 3.14159265358979323846;
	for( int j = 0; j <= nv; ++j ) {
		const double theta = pi * ( double )j / ( double )nv;
		for( int i = 0; i <= nu; ++i ) {
			const double phi = 2.0 * pi * ( double )i / ( double )nu;
			const float	 x	 = ( float )( std::sin( theta ) * std::cos( phi ) );
			const float	 y	 = ( float )std::cos( theta );
			const float	 z	 = ( float )( std::sin( theta ) * std::sin( phi ) );
			verts.push_back( x );
			verts.push_back( y );
			verts.push_back( z );
			normals.push_back( x );
			normals.push_back( y );
			normals.push_back( z );
			pbr.push_back( 0.5f * x + 0.5f );
			pbr.push_back( 0.5f * y + 0.5f );
			pbr.push_back( 0.5f * z + 0.5f );
			pbr.push_back( 0.25f );
			pbr.push_back( 0.75f );
			pbr.push_back( 1.0f );
		}
	}
	for( int j = 0; j < nv; ++j ) {
		for( int i = 0; i < nu; ++i ) {
			const int32_t a = j * ( nu + 1 ) + i;
			const int32_t b = a + 1;
			const int32_t c = a + ( nu + 1 );
			const int32_t d = c + 1;
			tris.push_back( a );
			tris.push_back( c );
			tris.push_back( d );
			tris.push_back( a );
			tris.push_back( d );
			tris.push_back( b );
		}
	}
}

} // namespace

int main()
{
	std::string err;

	std::printf( "projection backend: %s\n", t2print::projection_backend() );
	if( !t2print::projection_available() ) {
		std::fprintf( stderr, "project_pbr reports itself unavailable, which should be impossible\n" );
		return 1;
	}

	// ---------------------------------------------------------------------
	// Closest-surface transfer must use barycentric interpolation on the source
	// triangle, not nearest-vertex colors. The query is above (0.25,0.25,0),
	// whose expected RGB weights are (0.5,0.25,0.25).
	// ---------------------------------------------------------------------
	const float	  verts[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	const int32_t tris[]  = { 0, 1, 2 };
	const float	  pbr[]	  = {
		1,
		0,
		0,
		0,
		0.5f,
		1,
		0,
		1,
		0,
		0,
		0.5f,
		1,
		0,
		0,
		1,
		0,
		0.5f,
		1,
	};
	const std::vector<float>   source_verts( verts, verts + 9 );
	const std::vector<int32_t> source_tris( tris, tris + 3 );
	const std::vector<float>   source_pbr( pbr, pbr + 18 );
	const std::vector<float>   queries = { 0.25f, 0.25f, 0.5f };
	std::vector<float>		   projected;
	if( !t2print::project_pbr( source_verts, source_tris, source_pbr, queries, projected, err ) || projected.size() != 6 ) {
		std::fprintf( stderr, "project_pbr failed: %s\n", err.c_str() );
		return 1;
	}
	const float expected[] = { 0.5f, 0.25f, 0.25f, 0.0f, 0.5f, 1.0f };
	for( int i = 0; i < 6; ++i ) {
		if( std::fabs( projected[i] - expected[i] ) > 1e-5f ) {
			std::fprintf( stderr, "project_pbr channel %d: got %.7g expected %.7g\n", i, projected[i], expected[i] );
			return 1;
		}
	}

	// ---------------------------------------------------------------------
	// Degenerate and duplicated triangles. CGAL refuses degenerate primitives in
	// an AABB tree while tinybvh would happily index them, so both backends must
	// be fed the same filtered set — otherwise a backend comparison compares two
	// different inputs. A source that is *only* degenerate must fail cleanly
	// rather than return garbage.
	// ---------------------------------------------------------------------
	{
		std::vector<float>	 dv( source_verts );
		std::vector<int32_t> dt( source_tris );
		std::vector<float>	 dp( source_pbr );
		dv.insert( dv.end(), { 2, 0, 0, 3, 0, 0, 4, 0, 0 } ); // collinear
		dp.insert( dp.end(), 18, 0.0f );
		dt.insert( dt.end(), { 3, 4, 5 } ); // zero-area triangle
		dt.insert( dt.end(), { 0, 1, 2 } ); // exact duplicate of the first
		dt.insert( dt.end(), { 0, 0, 1 } ); // repeated index
		std::vector<float> got;
		if( !t2print::project_pbr( dv, dt, dp, queries, got, err ) || got.size() != 6 ) {
			std::fprintf( stderr, "project_pbr rejected a mesh with degenerate triangles: %s\n", err.c_str() );
			return 1;
		}
		for( int i = 0; i < 6; ++i ) {
			if( std::fabs( got[i] - expected[i] ) > 1e-5f ) {
				std::fprintf( stderr, "degenerate triangles perturbed the result: channel %d got %.7g expected %.7g\n", i, got[i], expected[i] );
				return 1;
			}
		}

		const std::vector<float>   only_verts = { 0, 0, 0, 1, 0, 0, 2, 0, 0 };
		const std::vector<int32_t> only_tris  = { 0, 1, 2 };
		const std::vector<float>   only_pbr( 18, 0.5f );
		std::vector<float>		   unused;
		if( t2print::project_pbr( only_verts, only_tris, only_pbr, queries, unused, err ) ) {
			std::fprintf( stderr, "an entirely degenerate source was accepted\n" );
			return 1;
		}
	}

	// ---------------------------------------------------------------------
	// Backend comparison. Both are compiled whenever CGAL is present, so this
	// runs them against each other on the same input in one binary.
	// ---------------------------------------------------------------------
	{
		std::vector<float>	 gv, gp;
		std::vector<int32_t> gt;
		make_grid( 32, gv, gt, gp );

		std::vector<float> q;
		for( uint32_t i = 0; i < 4096; ++i ) {
			q.push_back( unit( i * 3 + 0 ) * 2.4f - 1.2f );
			q.push_back( unit( i * 3 + 1 ) * 1.6f - 0.8f );
			q.push_back( unit( i * 3 + 2 ) * 2.4f - 1.2f );
		}

		std::vector<float> tiny;
		if( !t2print::project_pbr_backend( "tinybvh", gv, gt, gp, q, tiny, err ) ) {
			std::fprintf( stderr, "tinybvh projection failed: %s\n", err.c_str() );
			return 1;
		}
		if( tiny.size() != ( q.size() / 3 ) * 6 ) {
			std::fprintf( stderr, "tinybvh projection returned %zu values for %zu queries\n", tiny.size(), q.size() / 3 );
			return 1;
		}

		std::vector<float> cgal;
		if( t2print::project_pbr_backend( "cgal", gv, gt, gp, q, cgal, err ) ) {
			double worst	= 0.0;
			int	   worst_at = -1;
			for( size_t i = 0; i < tiny.size(); ++i ) {
				const double d = std::fabs( ( double )tiny[i] - cgal[i] );
				if( d > worst ) {
					worst	 = d;
					worst_at = ( int )i;
				}
			}
			// Both search the same filtered triangle set and interpolate with the
			// same code, so the legitimate difference is in where each lands on
			// the winning triangle: CGAL constructs the closest point with exact
			// predicates in double, tinybvh works in float, and the barycentric
			// weights inherit that. It is a small everywhere-present rounding
			// difference rather than the occasional shared-edge tie this comment
			// used to claim — the normal section below counts the affected
			// samples and finds roughly half of them. A deviation past this
			// bound is a different matter: it means the two searches disagree
			// about the nearest surface, which is a bug and not a tolerance to
			// widen.
			const double tolerance = 1e-4;
			std::printf( "backend agreement: max |tinybvh - cgal| = %.3g over %zu samples\n", worst, tiny.size() );
			if( worst > tolerance ) {
				std::fprintf( stderr, "backends disagree at value %d: tinybvh %.7g vs cgal %.7g (tolerance %.3g)\n", worst_at, tiny[worst_at], cgal[worst_at], tolerance );
				return 1;
			}
		} else {
			std::printf( "backend comparison skipped: %s\n", err.c_str() );
		}
	}

	// ---------------------------------------------------------------------
	// Shading normals off the same hit. This is what a tangent-space normal map
	// bake reads, so it is checked against a closed form (the unit sphere) and
	// not against another approximation.
	// ---------------------------------------------------------------------
	{
		std::vector<float>	 sv, sp, sn;
		std::vector<int32_t> st;
		make_sphere( 96, 48, sv, st, sp, sn );

		// Queries on a shell outside the sphere: the closest surface point is
		// essentially radial, so the exact normal there is the query direction.
		std::vector<float> q, dir;
		for( uint32_t i = 0; i < 4096; ++i ) {
			const float x = unit( i * 7 + 1 ) * 2.0f - 1.0f;
			const float y = unit( i * 7 + 3 ) * 2.0f - 1.0f;
			const float z = unit( i * 7 + 5 ) * 2.0f - 1.0f;
			const float l = std::sqrt( x * x + y * y + z * z );
			if( l < 1e-3f )
				continue; // no direction to speak of
			dir.push_back( x / l );
			dir.push_back( y / l );
			dir.push_back( z / l );
			q.push_back( 1.3f * x / l );
			q.push_back( 1.3f * y / l );
			q.push_back( 1.3f * z / l );
		}

		std::vector<float> pbr_ref;
		if( !t2print::project_pbr( sv, st, sp, q, pbr_ref, err ) ) {
			std::fprintf( stderr, "project_pbr on the sphere failed: %s\n", err.c_str() );
			return 1;
		}
		std::vector<float> pbr_with_n, nrm;
		if( !t2print::project_pbr_and_normals( nullptr, sv, st, sp, sn, q, pbr_with_n, nrm, err ) ) {
			std::fprintf( stderr, "project_pbr_and_normals failed: %s\n", err.c_str() );
			return 1;
		}
		if( nrm.size() != q.size() || pbr_with_n.size() != pbr_ref.size() ) {
			std::fprintf( stderr, "project_pbr_and_normals returned %zu normals / %zu pbr for %zu queries\n", nrm.size() / 3, pbr_with_n.size() / 6, q.size() / 3 );
			return 1;
		}
		// Asking for normals must not move the material by a single bit — the
		// colour bake and the normal bake read the very same hit.
		if( pbr_with_n != pbr_ref ) {
			std::fprintf( stderr, "requesting normals perturbed the PBR channels\n" );
			return 1;
		}

		// Judge normals by angle, not by raw component: for a unit vector the
		// angle is the quantity with a meaning, and it is what the bake reads.
		// The tessellation bound below is the closed-form budget — a 96x48
		// sphere has ~3.75 deg facets and the interpolated normal stays inside
		// that cone, so anything past a fraction of it means the wrong triangle
		// won the search or the weights are not the ones the material uses.
		const double pi_deg		 = 180.0 / 3.14159265358979323846;
		auto		 vs_analytic = [&]( const std::vector<float>& n, double& worst_deg, double& worst_len ) {
			worst_deg = 0.0;
			worst_len = 0.0;
			for( size_t i = 0; i < n.size() / 3; ++i ) {
				double dot = 0.0, len2 = 0.0;
				for( int k = 0; k < 3; ++k ) {
					dot += ( double )n[3 * i + k] * dir[3 * i + k];
					len2 += ( double )n[3 * i + k] * n[3 * i + k];
				}
				worst_len = std::max( worst_len, std::fabs( std::sqrt( len2 ) - 1.0 ) );
				worst_deg = std::max( worst_deg, std::acos( std::max( -1.0, std::min( 1.0, dot ) ) ) * pi_deg );
			}
		};
		double worst_deg = 0.0, worst_len = 0.0;
		vs_analytic( nrm, worst_deg, worst_len );
		std::printf( "sphere normals: worst %.4g deg off analytic, worst |len-1| = %.3g over %zu samples\n", worst_deg, worst_len, nrm.size() / 3 );
		if( worst_len > 1e-5 ) {
			std::fprintf( stderr, "projected normals are not unit length (worst |len-1| = %.3g)\n", worst_len );
			return 1;
		}
		if( worst_deg > 1.0 ) {
			std::fprintf( stderr, "projected normal deviates from the analytic sphere normal by %.4g deg\n", worst_deg );
			return 1;
		}

		// Cross-backend. Measured rather than assumed: the two disagree on
		// roughly half the samples, but by ~0.02 deg at worst. That is not the
		// shared-edge tie one might expect — it is everywhere-present, because
		// CGAL constructs the closest point with exact predicates in double
		// while tinybvh works in float, so the barycentric weights differ by
		// rounding across the whole field. (The material comparison further up
		// sees the same effect; it only ever looked at the maximum, which is
		// why it reads as a tie there.)
		//
		// So the count carries no signal and is reported, not asserted. What is
		// asserted is that *both* backends hit the analytic normal within the
		// same budget — a wrong-surface pick on a sphere would miss it by
		// degrees, which no rounding argument can absorb.
		std::vector<float> cgal_pbr, cgal_nrm;
		if( t2print::project_pbr_and_normals( "cgal", sv, st, sp, sn, q, cgal_pbr, cgal_nrm, err ) ) {
			double cgal_deg = 0.0, cgal_len = 0.0;
			vs_analytic( cgal_nrm, cgal_deg, cgal_len );
			const size_t samples	 = nrm.size() / 3;
			double		 worst_pair	 = 0.0;
			size_t		 disagreeing = 0;
			for( size_t i = 0; i < samples; ++i ) {
				double dot = 0.0;
				for( int k = 0; k < 3; ++k )
					dot += ( double )nrm[3 * i + k] * cgal_nrm[3 * i + k];
				const double deg = std::acos( std::max( -1.0, std::min( 1.0, dot ) ) ) * pi_deg;
				worst_pair		 = std::max( worst_pair, deg );
				if( deg > 1e-3 )
					++disagreeing;
			}
			std::printf( "backend normals: cgal %.4g deg off analytic; tinybvh vs cgal worst %.4g deg, %zu of %zu samples differ at all\n", cgal_deg, worst_pair, disagreeing, samples );
			if( cgal_len > 1e-5 || cgal_deg > 1.0 ) {
				std::fprintf( stderr, "the cgal backend's normals miss the analytic sphere by %.4g deg (|len-1| = %.3g)\n", cgal_deg, cgal_len );
				return 1;
			}
			if( worst_pair > 0.05 ) {
				std::fprintf( stderr, "backends disagree on the shading normal by %.4g deg, past a rounding difference\n", worst_pair );
				return 1;
			}
		} else {
			std::printf( "normal backend comparison skipped: %s\n", err.c_str() );
		}

		// A normal array that does not cover every source vertex is a caller bug
		// and must be refused, not silently indexed past the end.
		std::vector<float> short_normals( sn.begin(), sn.end() - 3 );
		std::vector<float> unused_pbr, unused_nrm;
		if( t2print::project_pbr_and_normals( nullptr, sv, st, sp, short_normals, q, unused_pbr, unused_nrm, err ) ) {
			std::fprintf( stderr, "a truncated source normal array was accepted\n" );
			return 1;
		}
	}

	// ---------------------------------------------------------------------
	// The unwrap -> per-texel source projection -> PBR PNG GLB path. No longer
	// CGAL-bound, so it runs in every build.
	// ---------------------------------------------------------------------
	t2glb::MeshExportOptions opt;
	opt.components	 = t2glb::ComponentFilter::KeepAll;
	opt.texture_size = 64;
	opt.dilate		 = 2;
	{
		// A two-triangle quad slightly above the source triangle: enough area to
		// unwrap, and every texel projects onto the source.
		const float			 target_verts[] = { -0.1f, -0.1f, 0.05f, 1.1f, -0.1f, 0.05f, 1.1f, 1.1f, 0.05f, -0.1f, 1.1f, 0.05f };
		const int32_t		 target_tris[]	= { 0, 1, 2, 0, 2, 3 };
		std::vector<uint8_t> glb;
		if( !t2glb::mesh_to_projected_glb( target_verts, 4, target_tris, 2, verts, 3, tris, 1, pbr, opt, glb, err ) ) {
			std::fprintf( stderr, "mesh_to_projected_glb failed: %s\n", err.c_str() );
			return 1;
		}
		if( glb.size() < 20 || std::memcmp( glb.data(), "glTF", 4 ) != 0 ) {
			std::fprintf( stderr, "projected bake returned an invalid GLB\n" );
			return 1;
		}
		const std::string glb_bytes( ( const char* )glb.data(), glb.size() );
		if( glb_bytes.find( "\"baseColorTexture\"" ) == std::string::npos || glb_bytes.find( "\"metallicRoughnessTexture\"" ) == std::string::npos ||
			glb_bytes.find( "\"TEXCOORD_0\"" ) == std::string::npos || glb_bytes.find( "\"COLOR_0\"" ) != std::string::npos ) {
			std::fprintf( stderr, "projected GLB is missing its UV PBR textures\n" );
			return 1;
		}
	}

	// ---------------------------------------------------------------------
	// Alpha Wrap is still CGAL-only. Everything above already ran, so a build
	// without CGAL reports the gap rather than masking it as a pass.
	// ---------------------------------------------------------------------
	if( !t2glb::print_remesh_available() ) {
		std::printf( "RESULT: PASS (projection only; Alpha Wrap section skipped, built without CGAL)\n" );
		return 0;
	}

	// A single open, zero-thickness triangle is deliberately not printable.
	// Alpha Wrap must enclose it in a closed volume despite having no usable
	// source connectivity or inside/outside orientation.
	t2glb::PreparedMesh out;
	if( !t2glb::prepare_print_mesh( verts, 3, tris, 1, pbr, opt, 0.20f, 0.03f, out, err ) ) {
		std::fprintf( stderr, "prepare_print_mesh failed: %s\n", err.c_str() );
		return 1;
	}
	if( out.verts.empty() || out.tris.empty() || out.normals.size() != out.verts.size() ) {
		std::fprintf( stderr, "empty/incomplete wrap: %zu verts %zu tris %zu normals\n", out.verts.size() / 3, out.tris.size() / 3, out.normals.size() / 3 );
		return 1;
	}
	// Alpha Wrap builds entirely new vertices, so source per-vertex material
	// cannot be carried over directly; prepare_print_mesh samples it onto them
	// by closest-surface projection for the preview. Every wrap vertex must get
	// a value, and the source's constant metal/rough/alpha must survive the
	// transfer unchanged.
	if( out.pbr.size() != out.verts.size() / 3 * 6 ) {
		std::fprintf( stderr, "preview PBR not projected onto every wrap vertex: %zu vs %zu\n", out.pbr.size() / 6, out.verts.size() / 3 );
		return 1;
	}
	for( size_t v = 0; v < out.pbr.size(); v += 6 ) {
		if( std::fabs( out.pbr[v + 3] - 0.0f ) > 1e-5f || std::fabs( out.pbr[v + 4] - 0.5f ) > 1e-5f || std::fabs( out.pbr[v + 5] - 1.0f ) > 1e-5f ) {
			std::fprintf( stderr, "projected preview material at vertex %zu: %g %g %g\n", v / 6, out.pbr[v + 3], out.pbr[v + 4], out.pbr[v + 5] );
			return 1;
		}
	}

	std::unordered_map<uint64_t, EdgeUse> edges;
	double								  signed_volume_6 = 0.0;
	for( size_t t = 0; t < out.tris.size(); t += 3 ) {
		const int32_t a = out.tris[t], b = out.tris[t + 1], c = out.tris[t + 2];
		for( const auto e : { std::pair<int32_t, int32_t> { a, b }, { b, c }, { c, a } } ) {
			const uint32_t lo  = ( uint32_t )std::min( e.first, e.second );
			const uint32_t hi  = ( uint32_t )std::max( e.first, e.second );
			EdgeUse&	   use = edges[( ( uint64_t )lo << 32 ) | hi];
			use.count++;
			use.direction += e.first < e.second ? 1 : -1;
		}
		const float* va = out.verts.data() + ( size_t )a * 3;
		const float* vb = out.verts.data() + ( size_t )b * 3;
		const float* vc = out.verts.data() + ( size_t )c * 3;
		signed_volume_6 += va[0] * ( vb[1] * vc[2] - vb[2] * vc[1] ) + va[1] * ( vb[2] * vc[0] - vb[0] * vc[2] ) + va[2] * ( vb[0] * vc[1] - vb[1] * vc[0] );
	}
	for( const auto& it : edges ) {
		if( it.second.count != 2 || it.second.direction != 0 ) {
			std::fprintf( stderr, "non-manifold/inconsistently wound edge: count=%d direction=%d\n", it.second.count, it.second.direction );
			return 1;
		}
	}
	if( std::fabs( signed_volume_6 ) < 1e-8 ) {
		std::fprintf( stderr, "Alpha Wrap did not enclose a non-zero volume\n" );
		return 1;
	}

	std::printf( "RESULT: PASS (%zu verts, %zu tris)\n", out.verts.size() / 3, out.tris.size() / 3 );
	return 0;
}
