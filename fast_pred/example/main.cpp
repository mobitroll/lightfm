#include <stdio.h>
#include "fastlightfm_api.h"
#include <string>


using namespace std;

void test(const string dir) {
    FastLFm fm = fast_lm_create();
    fast_lm_initialize(fm, dir.c_str());
    fast_lm_finalize(fm);
}

int main(int argc, const char* argv[]) {
    printf("start:\n");
    if (argc > 1) {
        test(string(argv[1]));
    }
}
