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
// @author, Yurii Shyrma (iuriish@yahoo.com), created on 06.12.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_invert_permutation)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(invert_permutation, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->isVector(), 0, "INVERT_PERMUTATION op: input array must be vector, but got shape %s instead !",
               ShapeUtils::shapeAsString(input).c_str());

  helpers::invertPermutation(block.launchContext(), *input, *output);

  return sd::Status::OK;
}

DECLARE_SYN(InvertPermutation, invert_permutation);

DECLARE_TYPES(invert_permutation) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setSameMode(true); }
}  // namespace ops
}  // namespace sd

#endif
