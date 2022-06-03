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
// Created by raver on 5/13/2018.
//
#include <array/NDArray.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/CustomOperations.h>

#include <fstream>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class MmapTests : public testing::Test {
 public:
};

TEST_F(MmapTests, Test_Basic_Mmap_1) {
  // FIXME: we must adopt this for CUDA as well
  if (!Environment::getInstance().isCPU()) return;

  // just 10GB
  sd::LongType size = 100000L;

  std::ofstream ofs("file", std::ios::binary | std::ios::out);
  ofs.seekp(size + 1024L);
  ofs.write("", 1);
  ofs.close();

  auto result = mmapFile(nullptr, "file", size);

  ASSERT_FALSE(result == nullptr);

  munmapFile(nullptr, result, size);

  remove("file");
}