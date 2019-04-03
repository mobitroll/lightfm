#ifndef FASTLIGHTFM_H
#define FASTLIGHTFM_H

#include "config.h"

#include <string>

#include "cnpy.h"
#include "csrmatrix.h"

/**
 * Class holds all the model state.
 */
class FastLightFM {
  CSRMatrix *item_features;
  CSRMatrix *user_features;

  cnpy::NpyArray item_embeddings;
  cnpy::NpyArray item_biases;

  cnpy::NpyArray user_embeddings;
  cnpy::NpyArray user_biases;

  int no_components;

  double item_scale;
  double user_scale;

  struct FastLightFMCache *lightfm_cache;

public:
  FastLightFM()
      : item_features(nullptr), user_features(nullptr), no_components(0),
        item_scale(1.0), user_scale(1.0), lightfm_cache(nullptr) {}

  virtual ~FastLightFM();

  void load(std::string dir);

  void predict(int user_id, int *item_ids, double *predictions,
               int no_examples, long *top_k_indice, long top_k);

  bool is_initialized();

#ifdef DEBUG
  void dump();
#endif // DEBUG

  CATCH_TEST
};

#endif // FASTLIGHTFM_H
