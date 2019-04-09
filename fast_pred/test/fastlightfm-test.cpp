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

TEST_CASE( "FastLightFM can warm-start predict (user 0)" ) {
    FastLightFM lfm;
    lfm.load("test/data/models/simple");

    long top_k = 4;
    int number_of_items = 5;
    int user_id_to_predict = 0;
    int item_ids_to_predict[] = { 0, 1, 2, 3, 4 };
    double predictions[number_of_items];
    long top_k_indices[top_k];

    lfm.predict(user_id_to_predict, item_ids_to_predict, predictions,
                number_of_items, top_k_indices, top_k);

    REQUIRE( predictions[0] != 0.0 );

    /* Expected ranking from Python version:
        {"item_id": "0", "score": 0.42357999086380005}, 
        {"item_id": "1", "score": -0.002106798579916358}, 
        {"item_id": "2", "score": -0.3895217776298523}, 
        {"item_id": "3", "score": -0.5090193152427673}, 
        {"item_id": "4", "score": -0.8530939817428589} */

    REQUIRE( top_k_indices[0] == 0 );
    REQUIRE( top_k_indices[1] == 1 );
    REQUIRE( top_k_indices[2] == 2 );
    REQUIRE( top_k_indices[3] == 3 );
}

TEST_CASE( "FastLightFM can warm-start predict (user 1)" ) {
    FastLightFM lfm;
    lfm.load("test/data/models/simple");

    long top_k = 4;
    int number_of_items = 5;
    int user_id_to_predict = 1;
    int item_ids_to_predict[] = { 0, 1, 2, 3, 4 };
    double predictions[number_of_items];
    long top_k_indices[top_k];

    lfm.predict(user_id_to_predict, item_ids_to_predict, predictions,
                number_of_items, top_k_indices, top_k);

    REQUIRE( predictions[0] != 0.0 );

    /* Expected ranking from Python version:
        {"item_id": "0", "score": 0.4042210280895233}, 
        {"item_id": "1", "score": -0.009759305976331234}, 
        {"item_id": "2", "score": -0.39128297567367554}, 
        {"item_id": "3", "score": -0.5066404938697815}, 
        {"item_id": "4", "score": -0.8442915081977844} */

    REQUIRE( top_k_indices[0] == 0 );
    REQUIRE( top_k_indices[1] == 1 );
    REQUIRE( top_k_indices[2] == 2 );
    REQUIRE( top_k_indices[3] == 3 );
}

TEST_CASE( "FastLightFM can warm-start predict (user 3)" ) {
    FastLightFM lfm;
    lfm.load("test/data/models/simple");

    long top_k = 4;
    int number_of_items = 5;
    int user_id_to_predict = 3;
    int item_ids_to_predict[] = { 0, 1, 2, 3, 4 };
    double predictions[number_of_items];
    long top_k_indices[top_k];

    lfm.predict(user_id_to_predict, item_ids_to_predict, predictions,
                number_of_items, top_k_indices, top_k);

    REQUIRE( predictions[0] != 0.0 );

    /* Expected ranking from Python version:
        {"item_id": "0", "score": 0.012117372825741768}, 
        {"item_id": "1", "score": -0.17272073030471802}, 
        {"item_id": "2", "score": -0.31047311425209045}, 
        {"item_id": "3", "score": -0.3740798532962799}, 
        {"item_id": "4", "score": -0.562594473361969} */

    REQUIRE( top_k_indices[0] == 0 );
    REQUIRE( top_k_indices[1] == 1 );
    REQUIRE( top_k_indices[2] == 2 );
    REQUIRE( top_k_indices[3] == 3 );
}
