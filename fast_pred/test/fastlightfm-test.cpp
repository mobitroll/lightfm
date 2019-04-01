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
    REQUIRE( lfm.item_features->rows == 5 );
    REQUIRE( lfm.user_features->rows == 7 );
    REQUIRE( lfm.is_initialized() );
}

TEST_CASE( "FastLightFM can warm-start predict" ) {
    FastLightFM lfm;
    lfm.load("test/data/models/simple");

    long top_k = 3;
    int number_of_users = 1;
    int number_of_items = 7;
    int user_ids_to_predict[] = { 0, 0, 0, 0, 0, 0, 0 };
    int item_ids_to_predict[] = { 0, 1, 2, 3, 4, 5, 6 };
    double predictions[number_of_items];
    long top_k_indices[top_k];

    lfm.predict(user_ids_to_predict, item_ids_to_predict, predictions, 
                number_of_users, top_k_indices, top_k);

    REQUIRE( predictions[0] != 0.0 );
}