#include "csrmatrix.h"

CSRMatrix* CSRMatrix::newInstance(cnpy::npz_t csrmatrix) {
  cnpy::NpyArray indices = csrmatrix["indices"];
  cnpy::NpyArray indptr = csrmatrix["indptr"];
  cnpy::NpyArray data = csrmatrix["data"];
  cnpy::NpyArray shape = csrmatrix["shape"];
  return new CSRMatrix(indices, indptr, data, shape);
}

CSRMatrix::CSRMatrix(cnpy::NpyArray &indices, cnpy::NpyArray &indptr,
        cnpy::NpyArray &data, cnpy::NpyArray &shape) {
    this->indices = indices;
    this->indptr = indptr;
    this->data = data;
    this->rows = shape.data<size_t>()[0];
    this->cols = shape.data<size_t>()[1];
    this->nnz = data.num_vals;
}

std::string CSRMatrix::toString() {
    std::string ret;
    ret += "indices:" + indices.toString() +
        "indptr:" + indptr.toString() +
        "data:" + data.toString() +
        "rows:" + lm_utils::intToString(this->rows) +
        " cols:" + lm_utils::intToString(this->cols);
    return ret;
}
