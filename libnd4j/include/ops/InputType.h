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
//

#ifndef ND4J_INPUTTYPE_H
#define ND4J_INPUTTYPE_H

namespace sd {
namespace ops {
enum InputType {
  InputType_BOOLEAN = 0,
  InputType_NUMERIC = 1,
  InputType_STRINGULAR = 2,
  InputType_NUMERIC_SET = 3,
  InputType_STRINGULAR_SET = 4,
};
}
}  // namespace sd

#endif
