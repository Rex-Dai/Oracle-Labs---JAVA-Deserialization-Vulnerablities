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
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYREDUCE_SAME_OP_H
#define LIBND4J_LEGACYREDUCE_SAME_OP_H
#include <ops/declarable/LegacyOp.h>

namespace sd {
namespace ops {
class SD_LIB_EXPORT LegacyReduceSameOp : public LegacyOp {
 protected:
  sd::Status validateAndExecute(Context& block) override;

 public:
  LegacyReduceSameOp();
  LegacyReduceSameOp(int opNum);

  ShapeList* calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block) override;
  LegacyOp* clone() override;
};
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LEGACYREDUCEOP_H
