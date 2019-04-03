#ifndef FASTLIGHTFM_API_H
#define FASTLIGHTFM_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* FastLFm;

const int FAST_LFM_OK            = 0;
const int FAST_LFM_ERROR_NULL_FM = 1;
const int FAST_LFM_ERROR_CANNOT_LOAD_MODEL = 2;

FastLFm fast_lfm_create();

int fast_lfm_initialize(FastLFm fm, const char * model_dir);

int fast_lfm_predict(FastLFm fm, int *user_ids, int *item_ids, double *predictions,
    int no_examples, long *top_k_indice, long top_k);

int fast_lfm_finalize(FastLFm fm);


#ifdef __cplusplus
}
#endif

#endif // FASTLIGHTFM_API_H
