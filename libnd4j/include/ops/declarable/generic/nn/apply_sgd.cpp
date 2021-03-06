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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_apply_sgd)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/gradient.h>

namespace sd {
namespace ops {
CONFIGURABLE_OP_IMPL(apply_sgd, 2, 1, true, -2, 0) {
  auto parameters = INPUT_VARIABLE(0);
  auto gradients = INPUT_VARIABLE(1);

  double lr = 0.0;

  REQUIRE_TRUE(
      parameters->isSameShape(gradients), 0,
      "ApplySGD: parameters and gradients should have the same shape, but got parameters = %s and gradients = %s !",
      ShapeUtils::shapeAsString(parameters).c_str(), ShapeUtils::shapeAsString(gradients).c_str());

  if (block.width() == 3) {
    auto tarr = INPUT_VARIABLE(2);
    lr = tarr->e<double>(0);
  } else if (block.getTArguments()->size() == 1) {
    lr = T_ARG(0);
  } else {
    REQUIRE_TRUE(false, 0, "ApplyGradients op should have LR announced either es T argument or additional NDArray!");
  }

  auto Z = OUTPUT_VARIABLE(0);

  helpers::applyGradientDescent(block.launchContext(), parameters, gradients, lr, Z);

  return sd::Status::OK;
}
DECLARE_SYN(ApplyGradientDescent, apply_sgd);
}  // namespace ops

DECLARE_TYPES(apply_sgd) { getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS}); }
}  // namespace sd

#endif
