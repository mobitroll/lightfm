#ifndef LM_UTILS_H
# define LM_UTILS_H

#include <string>

namespace lm_utils {

inline std::string intToString(int i) {
    char buff[256];
    snprintf(buff, 256, "%d", i);
    return std::string(buff);
}

}

#endif //LM_UTILS_H
