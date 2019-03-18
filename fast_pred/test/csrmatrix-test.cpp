#include "catch.hpp"
#include "fastlightfm.h"

/* See gennpz.py for the test data */

TEST_CASE( "Can load a csr sparse npz into CSRMatrix" ) {
    CSRMatrix *csrmatrix = loadCSRMatrix(cnpy::npz_load("test/data/sptest.npz"));
    REQUIRE( csrmatrix->rows == 3 );
    REQUIRE( csrmatrix->nnz == 3 );
    REQUIRE( csrmatrix->cols == 3 );
}

TEST_CASE( "Can load numpy array from npz" ) {
    cnpy::npz_t data = cnpy::npz_load("test/data/nptest.npz");
    cnpy::NpyArray arr = data["a"];
    REQUIRE( arr.shape.size() == 2 );
    REQUIRE( arr.shape[0] == 3 );
    REQUIRE( arr.shape[1] == 3 );
}