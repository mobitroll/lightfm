#include <stdio.h>
#include "fastlightfm.h"
#include <string>


using namespace std;

void test(const string dir) {
    auto fastlgm = new FastLightFM();
    fastlgm->load(dir);
    fastlgm->dump();
    delete fastlgm;
}


int main(int argc, const char* argv[]) {
    printf("start:\n");
    if (argc > 1) {
        test(string(argv[1]));
    }
}
