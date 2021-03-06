/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/NDArray.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void adjustSaturation(sd::LaunchContext* context, const NDArray* input, const NDArray* factorScalarArr,
                                    NDArray* output, const int dimC);

/*
    template <typename T>
    static SD_INLINE SD_HOST_DEVICE void rgb_to_hsv(T r, T g, T b, T* h, T* s, T* v) {
        T vv = sd::math::sd_max<T>(r, sd::math::sd_max<T>(g, b));
        T range = vv - sd::math::sd_min<T>(r, sd::math::sd_min<T>(g, b));
        if (vv > 0) {
            *s = range / vv;
        } else {
            *s = 0;
        }
        T norm = 1.0f / (6.0f * range);
        T hh;
        if (r == vv) {
            hh = norm * (g - b);
        } else if (g == vv) {
            hh = norm * (b - r) + 2.0 / 6.0;
        } else {
            hh = norm * (r - g) + 4.0 / 6.0;
        }
        if (range <= (T) 0.0) {
            hh = 0;
        }
        if (hh < (T) 0.0) {
            hh = hh + 1.;
        }
        *v = vv;
        *h = hh;
    }

    template <typename T>
    static SD_INLINE SD_HOST_DEVICE void hsv_to_rgb(T h, T s, T v, T* r, T* g, T* b) {
        T c = s * v;
        T m = v - c;
        T dh = h * 6;
        T rr, gg, bb;
        int h_category = static_cast<int>(dh);
        T fmodu = dh;
        while (fmodu <= (T) 0)
            fmodu += (T) 2.0f;

        while (fmodu >= (T) 2.0f)
            fmodu -= (T) 2.0f;

        T x = c * (1. - sd::math::sd_abs<T>(fmodu - 1.));
        switch (h_category) {
            case 0:
                rr = c;
                gg = x;
                bb = 0;
                break;
            case 1:
                rr = x;
                gg = c;
                bb = 0;
                break;
            case 2:
                rr = 0;
                gg = c;
                bb = x;
                break;
            case 3:
                rr = 0;
                gg = x;
                bb = c;
                break;
            case 4:
                rr = x;
                gg = 0;
                bb = c;
                break;
            case 5:
                rr = c;
                gg = 0;
                bb = x;
                break;
            default:
                rr = 0;
                gg = 0;
                bb = 0;
        }

        *r = rr + m;
        *g = gg + m;
        *b = bb + m;
    }
*/

}  // namespace helpers
}  // namespace ops
}  // namespace sd
