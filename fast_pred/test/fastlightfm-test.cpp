#include "catch.h"
#include "fastlightfm.h"

TEST_CASE( "FastLightFM can be constructed with defaults" ) {
    FastLightFM lfm;
    REQUIRE( !lfm.is_initialized() );
}

TEST_CASE( "FastLightFM can load model" ) {
    FastLightFM lfm;
    lfm.load("test/data/models/simple");
    REQUIRE( lfm.no_components == 10 );
    REQUIRE( lfm.is_initialized() );
}
