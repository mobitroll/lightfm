#include "utils.h"

CSRMatrix* CSRMatrix::newInstance(cnpy::npz_t csrmatrix) {
  cnpy::NpyArray indices = csrmatrix["indices"];
  cnpy::NpyArray indptr = csrmatrix["indptr"];
  cnpy::NpyArray data = csrmatrix["data"];
  cnpy::NpyArray shape = csrmatrix["shape"];
  return new CSRMatrix(indices, indptr, data, shape);
}
