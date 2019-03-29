#ifndef CSRMATRIX_H
# define CSRMATRIX_H

#include "config.h"
#include <string>
#include "cnpy.h"
#include "utils.h"

/*
 * Utility class for accessing elements
 * of a CSR matrix.
 */
struct CSRMatrix {
  cnpy::NpyArray indices;
  cnpy::NpyArray indptr;
  cnpy::NpyArray data;

  int rows, cols, nnz;

  CSRMatrix(cnpy::NpyArray &indices, cnpy::NpyArray &indptr,
            cnpy::NpyArray &data, cnpy::NpyArray &shape);

  std::string toString();

  /*
   * Return the pointer to the start of the
   * data for row.
   */
  int get_row_start(int row) { return indptr.data<int>()[row]; }

  /*
   *  Return the pointer to the end of the
   *  data for row.
   */
  int get_row_end(int row) { return indptr.data<int>()[row + 1]; }

  static CSRMatrix* newInstance(cnpy::npz_t csrmatrix);

  CATCH_TEST
};

#endif // CSRMATRIX_H
