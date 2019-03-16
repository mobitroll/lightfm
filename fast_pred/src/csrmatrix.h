#ifndef CSRMATRIX_H
# define CSRMATRIX_H

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
            cnpy::NpyArray &data, cnpy::NpyArray &shape) {
    this->indices = indices;
    this->indptr = indptr;
    this->data = data;
    this->rows = shape.data<size_t>()[0];
    this->cols = shape.data<size_t>()[1];
    this->nnz = data.num_vals;
  }

  std::string toString() {
    std::string ret;
    ret += "indices:" + indices.toString() +
           "indptr:" + indptr.toString() +
           "data:" + data.toString() +
           "rows:" + lm_utils::intToString(this->rows) +
           "cols:" + lm_utils::intToString(this->cols);
    return ret;
  }

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
};

#endif // CSRMATRIX_H
