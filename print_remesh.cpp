#include "print_remesh.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>

#ifdef TRELLIS2_USE_CGAL

	#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
	#include <CGAL/AABB_tree.h>
	#if __has_include( <CGAL/AABB_traits_3.h>)
		#include <CGAL/AABB_traits_3.h>
		#include <CGAL/AABB_triangle_primitive_3.h>
		#define T2_CGAL_AABB_3_NAMES 1
	#else
		#include <CGAL/AABB_traits.h>
		#include <CGAL/AABB_triangle_primitive.h>
	#endif
	#include <CGAL/Surface_mesh.h>
	#include <CGAL/alpha_wrap_3.h>

	#include <array>

#endif

namespace t2print
{

bool available()
{
#ifdef TRELLIS2_USE_CGAL
	return true;
#else
	return false;
#endif
}

namespace
{

	void vertex_normals( const std::vector<float>& verts, const std::vector<int32_t>& tris, std::vector<float>& normals )
	{
		normals.assign( verts.size(), 0.0f );
		for( size_t t = 0; t + 2 < tris.size(); t += 3 ) {
			const int32_t ia = tris[t], ib = tris[t + 1], ic = tris[t + 2];
			const float*  a		= verts.data() + ( size_t )ia * 3;
			const float*  b		= verts.data() + ( size_t )ib * 3;
			const float*  c		= verts.data() + ( size_t )ic * 3;
			const float	  ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
			const float	  ac[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
			const float	  n[3]	= {
				   ab[1] * ac[2] - ab[2] * ac[1],
				   ab[2] * ac[0] - ab[0] * ac[2],
				   ab[0] * ac[1] - ab[1] * ac[0],
			};
			for( int32_t i : { ia, ib, ic } ) {
				normals[( size_t )i * 3 + 0] += n[0];
				normals[( size_t )i * 3 + 1] += n[1];
				normals[( size_t )i * 3 + 2] += n[2];
			}
		}
		for( size_t i = 0; i < verts.size() / 3; ++i ) {
			float*		n	= normals.data() + i * 3;
			const float len = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
			if( len > 1e-20f ) {
				n[0] /= len;
				n[1] /= len;
				n[2] /= len;
			}
		}
	}

} // namespace

bool alpha_wrap( const std::vector<float>& source_verts,
	const std::vector<int32_t>&			   source_tris,
	float								   alpha_ratio,
	float								   offset_ratio,
	std::vector<float>&					   out_verts,
	std::vector<float>&					   out_normals,
	std::vector<int32_t>&				   out_tris,
	std::string&						   err )
{
	out_verts.clear();
	out_normals.clear();
	out_tris.clear();
#ifndef TRELLIS2_USE_CGAL
	( void )source_verts;
	( void )source_tris;
	( void )alpha_ratio;
	( void )offset_ratio;
	err = "print remeshing is unavailable (rebuild with CGAL >= 5.5)";
	return false;
#else
	if( source_verts.size() < 9 || source_tris.size() < 3 ) {
		err = "empty mesh";
		return false;
	}
	if( !std::isfinite( alpha_ratio ) || !std::isfinite( offset_ratio ) || alpha_ratio <= 0.0f || alpha_ratio > 0.5f || offset_ratio <= 0.0f || offset_ratio > 0.5f ) {
		err = "bad Alpha Wrap parameters";
		return false;
	}

	using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
	using Point	 = Kernel::Point_3;
	using Mesh	 = CGAL::Surface_mesh<Point>;

	try {
		std::vector<Point> points;
		points.reserve( source_verts.size() / 3 );
		double lo[3] = { 1e300, 1e300, 1e300 };
		double hi[3] = { -1e300, -1e300, -1e300 };
		for( size_t i = 0; i < source_verts.size() / 3; ++i ) {
			double p[3] = { source_verts[3 * i], source_verts[3 * i + 1], source_verts[3 * i + 2] };
			if( !std::isfinite( p[0] ) || !std::isfinite( p[1] ) || !std::isfinite( p[2] ) ) {
				err = "mesh contains a non-finite vertex";
				return false;
			}
			points.emplace_back( p[0], p[1], p[2] );
			for( int k = 0; k < 3; ++k ) {
				lo[k] = std::min( lo[k], p[k] );
				hi[k] = std::max( hi[k], p[k] );
			}
		}

		std::vector<std::array<std::size_t, 3>> faces;
		faces.reserve( source_tris.size() / 3 );
		for( size_t t = 0; t < source_tris.size() / 3; ++t ) {
			const int32_t a = source_tris[3 * t], b = source_tris[3 * t + 1], c = source_tris[3 * t + 2];
			if( a < 0 || b < 0 || c < 0 || ( size_t )a >= points.size() || ( size_t )b >= points.size() || ( size_t )c >= points.size() ) {
				err = "triangle index out of range";
				return false;
			}
			if( a == b || b == c || a == c )
				continue;
			faces.push_back( { { ( size_t )a, ( size_t )b, ( size_t )c } } );
		}
		if( faces.empty() ) {
			err = "mesh has no valid triangles";
			return false;
		}

		const double dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
		const double diagonal = std::sqrt( dx * dx + dy * dy + dz * dz );
		if( !( diagonal > 0.0 ) || !std::isfinite( diagonal ) ) {
			err = "mesh has an empty bounding box";
			return false;
		}

		Mesh wrap;
		CGAL::alpha_wrap_3( points, faces, diagonal * ( double )alpha_ratio, diagonal * ( double )offset_ratio, wrap );
		if( wrap.is_empty() || wrap.number_of_faces() == 0 ) {
			err = "CGAL Alpha Wrap produced an empty mesh";
			return false;
		}

		// Surface_mesh descriptors are indices but are not required to be
		// densely packed, so retain an explicit descriptor-to-output remap.
		std::vector<int32_t> remap;
		out_verts.reserve( wrap.number_of_vertices() * 3 );
		for( Mesh::Vertex_index v : wrap.vertices() ) {
			if( ( size_t )v.idx() >= remap.size() )
				remap.resize( ( size_t )v.idx() + 1, -1 );
			remap[v.idx()] = ( int32_t )( out_verts.size() / 3 );
			const Point& p = wrap.point( v );
			out_verts.push_back( ( float )CGAL::to_double( p.x() ) );
			out_verts.push_back( ( float )CGAL::to_double( p.y() ) );
			out_verts.push_back( ( float )CGAL::to_double( p.z() ) );
		}

		out_tris.reserve( wrap.number_of_faces() * 3 );
		for( Mesh::Face_index f : wrap.faces() ) {
			Mesh::Halfedge_index h = wrap.halfedge( f );
			for( int k = 0; k < 3; ++k ) {
				Mesh::Vertex_index v = wrap.target( h );
				if( ( size_t )v.idx() >= remap.size() || remap[v.idx()] < 0 ) {
					err = "CGAL Alpha Wrap returned an invalid face";
					return false;
				}
				out_tris.push_back( remap[v.idx()] );
				h = wrap.next( h );
			}
			if( h != wrap.halfedge( f ) ) {
				err = "CGAL Alpha Wrap returned a non-triangle face";
				return false;
			}
		}
		vertex_normals( out_verts, out_tris, out_normals );
		return true;
	} catch( const std::exception& ex ) {
		err = std::string( "CGAL Alpha Wrap failed: " ) + ex.what();
		return false;
	} catch( ... ) {
		err = "CGAL Alpha Wrap failed";
		return false;
	}
#endif
}

bool project_pbr( const std::vector<float>& source_verts,
	const std::vector<int32_t>&				source_tris,
	const std::vector<float>&				source_pbr,
	const std::vector<float>&				query_points,
	std::vector<float>&						out_pbr,
	std::string&							err )
{
	out_pbr.clear();
#ifndef TRELLIS2_USE_CGAL
	( void )source_verts;
	( void )source_tris;
	( void )source_pbr;
	( void )query_points;
	err = "PBR projection is unavailable (rebuild with CGAL >= 5.5)";
	return false;
#else
	using Kernel			= CGAL::Exact_predicates_inexact_constructions_kernel;
	using Point				= Kernel::Point_3;
	using Triangle			= Kernel::Triangle_3;
	using Triangle_iterator = std::vector<Triangle>::const_iterator;
	#ifdef T2_CGAL_AABB_3_NAMES
	using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, Triangle_iterator>;
	using Traits	= CGAL::AABB_traits_3<Kernel, Primitive>;
	#else
	// These compatibility names are used by CGAL 5.5, the first Alpha Wrap
	// release. CGAL 6 selects the non-deprecated aliases above.
	using Primitive = CGAL::AABB_triangle_primitive<Kernel, Triangle_iterator>;
	using Traits	= CGAL::AABB_traits<Kernel, Primitive>;
	#endif
	using Tree = CGAL::AABB_tree<Traits>;

	const size_t source_nv = source_verts.size() / 3;
	if( source_verts.size() < 9 || source_verts.size() % 3 != 0 || source_tris.size() < 3 || source_tris.size() % 3 != 0 ) {
		err = "empty PBR projection source";
		return false;
	}
	if( source_pbr.size() != source_nv * 6 ) {
		err = "PBR projection source has no six-channel material";
		return false;
	}
	if( query_points.size() % 3 != 0 ) {
		err = "PBR projection query array is not xyz-aligned";
		return false;
	}
	if( query_points.empty() )
		return true;

	try {
		std::vector<Triangle>				triangles;
		std::vector<std::array<int32_t, 3>> source_faces;
		triangles.reserve( source_tris.size() / 3 );
		source_faces.reserve( source_tris.size() / 3 );
		for( size_t t = 0; t < source_tris.size() / 3; ++t ) {
			const int32_t ia = source_tris[3 * t], ib = source_tris[3 * t + 1], ic = source_tris[3 * t + 2];
			if( ia < 0 || ib < 0 || ic < 0 || ( size_t )ia >= source_nv || ( size_t )ib >= source_nv || ( size_t )ic >= source_nv ) {
				err = "PBR projection triangle index out of range";
				return false;
			}
			const Point a( source_verts[3 * ( size_t )ia], source_verts[3 * ( size_t )ia + 1], source_verts[3 * ( size_t )ia + 2] );
			const Point b( source_verts[3 * ( size_t )ib], source_verts[3 * ( size_t )ib + 1], source_verts[3 * ( size_t )ib + 2] );
			const Point c( source_verts[3 * ( size_t )ic], source_verts[3 * ( size_t )ic + 1], source_verts[3 * ( size_t )ic + 2] );
			Triangle	tri( a, b, c );
			// CGAL explicitly disallows degenerate primitives in an AABB tree.
			if( tri.is_degenerate() )
				continue;
			triangles.push_back( tri );
			source_faces.push_back( { { ia, ib, ic } } );
		}
		if( triangles.empty() ) {
			err = "PBR projection source has no non-degenerate triangles";
			return false;
		}

		Tree tree( triangles.cbegin(), triangles.cend() );
		tree.build();
		tree.accelerate_distance_queries();
		out_pbr.resize( ( query_points.size() / 3 ) * 6 );

		auto sample_one = [&]( size_t qi ) {
			const Point	 query( query_points[3 * qi], query_points[3 * qi + 1], query_points[3 * qi + 2] );
			const auto	 hit = tree.closest_point_and_primitive( query );
			const size_t ti	 = ( size_t )std::distance( triangles.cbegin(), hit.second );
			const auto&	 ids = source_faces[ti];
			const Point& a	 = triangles[ti].vertex( 0 );
			const Point& b	 = triangles[ti].vertex( 1 );
			const Point& c	 = triangles[ti].vertex( 2 );
			const Point& q	 = hit.first;

			const double ab[3] = { CGAL::to_double( b.x() - a.x() ), CGAL::to_double( b.y() - a.y() ), CGAL::to_double( b.z() - a.z() ) };
			const double ac[3] = { CGAL::to_double( c.x() - a.x() ), CGAL::to_double( c.y() - a.y() ), CGAL::to_double( c.z() - a.z() ) };
			const double aq[3] = { CGAL::to_double( q.x() - a.x() ), CGAL::to_double( q.y() - a.y() ), CGAL::to_double( q.z() - a.z() ) };
			const double d00   = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
			const double d01   = ab[0] * ac[0] + ab[1] * ac[1] + ab[2] * ac[2];
			const double d11   = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
			const double d20   = aq[0] * ab[0] + aq[1] * ab[1] + aq[2] * ab[2];
			const double d21   = aq[0] * ac[0] + aq[1] * ac[1] + aq[2] * ac[2];
			const double denom = d00 * d11 - d01 * d01;
			double		 wb	   = ( d11 * d20 - d01 * d21 ) / denom;
			double		 wc	   = ( d00 * d21 - d01 * d20 ) / denom;
			double		 wa	   = 1.0 - wb - wc;
			// The closest point is on the triangle. Clamp only numerical noise
			// so interpolation remains stable on edges and vertices.
			wa				 = std::max( 0.0, std::min( 1.0, wa ) );
			wb				 = std::max( 0.0, std::min( 1.0, wb ) );
			wc				 = std::max( 0.0, std::min( 1.0, wc ) );
			const double sum = wa + wb + wc;
			wa /= sum;
			wb /= sum;
			wc /= sum;
			for( int ch = 0; ch < 6; ++ch ) {
				out_pbr[6 * qi + ( size_t )ch] =
					( float )( wa * source_pbr[6 * ( size_t )ids[0] + ( size_t )ch] + wb * source_pbr[6 * ( size_t )ids[1] + ( size_t )ch] + wc * source_pbr[6 * ( size_t )ids[2] + ( size_t )ch] );
			}
		};

		const size_t nq = query_points.size() / 3;
	#ifdef CGAL_HAS_THREADS
		const unsigned		hw		= std::max( 1u, std::thread::hardware_concurrency() );
		const unsigned		workers = ( unsigned )std::min<size_t>( std::min( 16u, hw ), ( nq + 4095 ) / 4096 );
		std::atomic<size_t> next { 0 };
		std::atomic<bool>	failed { false };
		std::mutex			failure_mu;
		std::string			failure;
		auto				worker = [&]() {
			   try {
				   for( ;; ) {
					   const size_t begin = next.fetch_add( 4096 );
					   if( begin >= nq || failed.load() )
						   break;
					   const size_t end = std::min( nq, begin + 4096 );
					   for( size_t qi = begin; qi < end; ++qi )
						   sample_one( qi );
				   }
			   } catch( const std::exception& ex ) {
				   failed.store( true );
				   std::lock_guard<std::mutex> lock( failure_mu );
				   if( failure.empty() )
					   failure = ex.what();
			   } catch( ... ) {
				   failed.store( true );
			   }
		};
		std::vector<std::thread> threads;
		threads.reserve( workers );
		for( unsigned i = 0; i < workers; ++i )
			threads.emplace_back( worker );
		for( auto& thread : threads )
			thread.join();
		if( failed.load() ) {
			err = failure.empty() ? "CGAL PBR projection failed" : std::string( "CGAL PBR projection failed: " ) + failure;
			out_pbr.clear();
			return false;
		}
	#else
		for( size_t qi = 0; qi < nq; ++qi )
			sample_one( qi );
	#endif
		return true;
	} catch( const std::exception& ex ) {
		err = std::string( "CGAL PBR projection failed: " ) + ex.what();
		out_pbr.clear();
		return false;
	} catch( ... ) {
		err = "CGAL PBR projection failed";
		out_pbr.clear();
		return false;
	}
#endif
}

} // namespace t2print
