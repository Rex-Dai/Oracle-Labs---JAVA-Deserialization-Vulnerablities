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
#if NOT_EXCLUDED(OP_get_seed)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(get_seed, -2, 1, false, 0, 0) {
  //            REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RNG should be defined in Graph");
  auto rng = block.getRng();
  auto z = OUTPUT_VARIABLE(0);

  z->p(sd::LongType(0), rng.rootState());

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(get_seed) {
  auto newshape = ConstantShapeHelper::getInstance().scalarShapeInfo(DataType::INT64);
  return SHAPELIST(newshape);
}

DECLARE_TYPES(get_seed) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes(DataType::INT64);
}
}  // namespace ops
}  // namespace sd

#endif
