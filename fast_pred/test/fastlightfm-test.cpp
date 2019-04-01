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

    long top_k = 5;
    int number_of_users = 5;
    int number_of_items = 5;
    int user_ids_to_predict[] = { 0, 0, 0, 0, 0 };
    int item_ids_to_predict[] = { 0, 1, 2, 3, 4 };
    double predictions[number_of_items];
    long top_k_indices[top_k];

    lfm.predict(user_ids_to_predict, item_ids_to_predict, predictions, 
                number_of_users, top_k_indices, top_k);

    REQUIRE( predictions[0] != 0.0 );

    for (int i = 0; i < top_k; i++) {
        std::cout << top_k_indices[i] << "\n";
    }

    for (int j = 0; j < number_of_items; j++) {
        std::cout << predictions[j] << "\n";
    }

    /* 
        a or b -> 0 or 1 // known
        a or b -> 0 or 1 // known
        e -> 4
        c -> 2
        d -> 3
    */

    REQUIRE( top_k_indices[2] == 4 );
    REQUIRE( top_k_indices[3] == 2 );
    REQUIRE( top_k_indices[4] == 3 );
}