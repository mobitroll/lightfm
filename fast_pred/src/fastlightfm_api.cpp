#include "fastlightfm_api.h"

#include "fastlightfm.h"

static FastLightFM*  get_fastlfm_obj(FastLFm fm) {
    return reinterpret_cast<FastLightFM*>(fm);
}

extern "C" FastLFm fast_lfm_create() {
    return new FastLightFM();
}

extern "C" int fast_lfm_initialize(FastLFm fm, const char * model_dir) {
    if (!fm) {
        return FAST_LFM_ERROR_NULL_FM;
    }
   
    auto fastlgm = get_fastlfm_obj(fm);
    fastlgm->load(model_dir);

    if (!fastlgm->is_initialized()) {
        fast_lfm_finalize(fastlgm);
        return FAST_LFM_ERROR_CANNOT_LOAD_MODEL;
    }

    return FAST_LFM_OK;
}

extern "C" int fast_lfm_predict(FastLFm fm, int user_id, int *item_ids, double *predictions,
               int no_examples, long *top_k_indice, long top_k) {
    if (!fm) {
        return FAST_LFM_ERROR_NULL_FM;
    }

    auto fastlgm = get_fastlfm_obj(fm);
    fastlgm->predict(user_id, item_ids, predictions, no_examples, top_k_indice, top_k);
    return FAST_LFM_OK;
}

extern "C" int fast_lfm_finalize(FastLFm fm) {
    delete get_fastlfm_obj(fm);
    return FAST_LFM_OK;
}
