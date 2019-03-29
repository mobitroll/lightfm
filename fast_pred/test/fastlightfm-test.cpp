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
    REQUIRE( lfm.number_of_items == 5 );
    REQUIRE( lfm.number_of_users == 7 );
    REQUIRE( lfm.is_initialized() );
}

TEST_CASE( "FastLightFM can warm-start predict for a single item" ) {
    FastLightFM lfm;
    lfm.load("test/data/models/simple");

    long top_k = 1;
    int no_examples = 1;
    int user_ids[] = { 0 };
    int item_ids[] = { 0 };
    double predictions[no_examples];
    long top_k_indice[top_k];

    lfm.predict(user_ids, item_ids, predictions, no_examples, top_k_indice, top_k);

    REQUIRE( predictions[0] != 0 );
}