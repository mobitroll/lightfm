#include "fastlightfm_api.h"

#include "fastlightfm.h"

static FastLightFM*  get_fastlfm_obj(FastLFm fm) {
    return reinterpret_cast<FastLightFM*>(fm);
}

extern "C" FastLFm fast_lm_create() {
    return new FastLightFM();
}

extern "C" int fast_lm_initialize(FastLFm fm, const char * model_dir) {
    if (!fm) {
        return FAST_LFM_ERROR_NULL_FM;
    }
   
    auto fastlgm = get_fastlfm_obj(fm);
    fastlgm->load(model_dir);

    if (!fastlgm->is_initialized()) {
        fast_lm_finalize(fastlgm);
        return FAST_LFM_ERROR_CANNOT_LOAD_MODEL;
    }

    return FAST_LFM_OK;
}

extern "C" int fast_lfm_predict(FastLFm fm, int *user_ids, int *item_ids, double *predictions,
               int no_examples, long *top_k_indice, long top_k) {
    if (!fm) {
        return FAST_LFM_ERROR_NULL_FM;
    }

    auto fastlgm = get_fastlfm_obj(fm);
    fastlgm->predict(user_ids, item_ids, predictions, no_examples, top_k_indice, top_k);
    return FAST_LFM_OK;
}

extern "C" int fast_lm_finalize(FastLFm fm) {
    delete get_fastlfm_obj(fm);
    return FAST_LFM_OK;
}
