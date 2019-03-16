#include "catch.hpp"
#include "fastlightfm.h"

TEST_CASE( "FastLightFM can be constructed with defaults" ) {
    FastLightFM lfm;
    REQUIRE( !lfm.is_initialized() );
}
