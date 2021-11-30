#include "array_utils.hpp"

double calc_pmerge(size_t len0, size_t len1, size_t union_count) {

    size_t len = len0 + len1;
    return (union_count) / static_cast<double>(len);
}

double calc_pmerge(double avglen0, double avglen1, double avgunions) {
    double avglen = avglen0 + avglen1;
    return (avgunions) / avglen;
}
