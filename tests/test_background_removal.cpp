#include "trellis2.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{

std::vector<uint8_t> solid( int w, int h, uint8_t v )
{
	std::vector<uint8_t> im( ( size_t )w * h * 4, 255 );
	for( int i = 0; i < w * h; ++i ) {
		im[( size_t )i * 4 + 0] = v;
		im[( size_t )i * 4 + 1] = v;
		im[( size_t )i * 4 + 2] = v;
	}
	return im;
}

void rgb( std::vector<uint8_t>& im, int w, int x, int y, uint8_t r, uint8_t g, uint8_t b )
{
	uint8_t* p = &im[( ( size_t )y * w + x ) * 4];
	p[0]	   = r;
	p[1]	   = g;
	p[2]	   = b;
	p[3]	   = 255;
}

uint8_t alpha( const std::vector<uint8_t>& im, int w, int x, int y )
{
	return im[( ( size_t )y * w + x ) * 4 + 3];
}

bool expect( bool condition, const char* message )
{
	if( !condition )
		std::fprintf( stderr, "%s\n", message );
	return condition;
}

} // namespace

int main()
{
	bool ok = true;

	// White and black backgrounds are detected from the border while a
	// disconnected contrasting subject remains opaque.
	for( int bg : { 0, 8, 247, 255 } ) {
		auto im = solid( 9, 9, ( uint8_t )bg );
		for( int y = 3; y <= 5; ++y )
			for( int x = 3; x <= 5; ++x ) {
				if( bg < 128 )
					rgb( im, 9, x, y, 235, 80, 40 );
				else
					rgb( im, 9, x, y, 20, 90, 180 );
			}
		const int changed = trellis2_remove_solid_background_rgba( im.data(), 9, 9, TRELLIS2_BACKGROUND_AUTO );
		ok &= expect( changed > 0, bg < 128 ? "black background not detected" : "white background not detected" );
		ok &= expect( alpha( im, 9, 0, 0 ) <= 1, "background corner is not transparent" );
		ok &= expect( alpha( im, 9, 4, 4 ) == 255, "subject centre lost opacity" );
	}

	// Border connectivity prevents an enclosed white detail from being erased
	// along with the surrounding white background.
	auto enclosed = solid( 9, 9, 255 );
	for( int y = 2; y <= 6; ++y )
		for( int x = 2; x <= 6; ++x )
			rgb( enclosed, 9, x, y, 180, 30, 20 );
	rgb( enclosed, 9, 4, 4, 255, 255, 255 );
	trellis2_remove_solid_background_rgba( enclosed.data(), 9, 9, TRELLIS2_BACKGROUND_AUTO );
	ok &= expect( alpha( enclosed, 9, 4, 4 ) == 255, "enclosed white subject detail was removed" );

	// A genuine existing alpha mask wins over automatic colour removal.
	auto masked = solid( 20, 20, 255 );
	for( int y = 0; y < 3; ++y )
		for( int x = 0; x < 3; ++x )
			masked[( ( size_t )y * 20 + x ) * 4 + 3] = 0;
	const int changed = trellis2_remove_solid_background_rgba( masked.data(), 20, 20, TRELLIS2_BACKGROUND_AUTO );
	ok &= expect( changed == 0, "existing alpha mask was unexpectedly modified" );
	ok &= expect( alpha( masked, 20, 10, 10 ) == 255, "existing mask made opaque content transparent" );

	// Mid-tone borders are not guessed as black or white in auto mode.
	auto grey = solid( 9, 9, 128 );
	ok &= expect( trellis2_remove_solid_background_rgba( grey.data(), 9, 9, TRELLIS2_BACKGROUND_AUTO ) == 0, "mid-grey background should be left unchanged" );

	// Explicit modes override auto detection but still affect border-connected
	// pixels only. This is useful when a subject occupies most of the border.
	auto forced = solid( 9, 9, 120 );
	for( int x = 0; x < 4; ++x )
		rgb( forced, 9, x, 0, 0, 0, 0 );
	ok &= expect( trellis2_remove_solid_background_rgba( forced.data(), 9, 9, TRELLIS2_BACKGROUND_BLACK ) > 0, "forced black mode did not remove a connected black region" );
	ok &= expect( alpha( forced, 9, 0, 0 ) == 0, "forced black pixel is not transparent" );
	ok &= expect( alpha( forced, 9, 8, 8 ) == 255, "forced mode modified unrelated pixels" );

	if( !ok )
		return 1;
	std::puts( "background removal: ok" );
	return 0;
}
