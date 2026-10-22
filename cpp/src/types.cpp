#include "types.h"

#include <algorithm>

namespace memecam {

Box bbox_of(const Vec3* pts, size_t n) {
    if (n == 0) return {};
    double x0 = pts[0][0], y0 = pts[0][1], x1 = x0, y1 = y0;
    for (size_t i = 1; i < n; ++i) {
        x0 = std::min(x0, pts[i][0]); y0 = std::min(y0, pts[i][1]);
        x1 = std::max(x1, pts[i][0]); y1 = std::max(y1, pts[i][1]);
    }
    return box_from_corners(int(x0), int(y0), int(x1), int(y1));
}

}  // namespace memecam
