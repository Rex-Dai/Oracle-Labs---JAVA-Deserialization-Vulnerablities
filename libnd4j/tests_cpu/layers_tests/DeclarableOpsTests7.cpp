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
// Created by raver119 on 09.02.18.
//

#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <helpers/GradCheck.h>
#include <helpers/helper_hash.h>
#include <ops/declarable/CustomOperations.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class DeclarableOpsTests7 : public testing::Test {
 public:
  DeclarableOpsTests7() {
    printf("\n");
    fflush(stdout);
  }
};

template <typename T>
class TypedDeclarableOpsTests7 : public testing::Test {
 public:
  TypedDeclarableOpsTests7() {
    printf("\n");
    fflush(stdout);
  }
};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedDeclarableOpsTests7, TestingTypes);

TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_LARGE) {
  double inputData[150] = {
      0,    0.51, 0.68, 0.69, 0.86, 0.91, 0.96, 0.97, 0.97, 1.03, 1.13, 1.16, 1.16, 1.17, 1.19, 1.25, 1.25, 1.26, 1.27,
      1.28, 1.29, 1.29, 1.29, 1.30, 1.31, 1.32, 1.33, 1.33, 1.35, 1.35, 1.36, 1.37, 1.38, 1.40, 1.41, 1.42, 1.43, 1.44,
      1.44, 1.45, 1.45, 1.47, 1.47, 1.51, 1.51, 1.51, 1.52, 1.53, 1.56, 1.57, 1.58, 1.59, 1.61, 1.62, 1.63, 1.63, 1.64,
      1.64, 1.66, 1.66, 1.67, 1.67, 1.70, 1.70, 1.70, 1.72, 1.72, 1.72, 1.72, 1.73, 1.74, 1.74, 1.76, 1.76, 1.77, 1.77,
      1.80, 1.80, 1.81, 1.82, 1.83, 1.83, 1.84, 1.84, 1.84, 1.85, 1.85, 1.85, 1.86, 1.86, 1.87, 1.88, 1.89, 1.89, 1.89,
      1.89, 1.89, 1.91, 1.91, 1.91, 1.92, 1.94, 1.95, 1.97, 1.98, 1.98, 1.98, 1.98, 1.98, 1.99, 2,    2,    2.01, 2.01,
      2.02, 2.03, 2.03, 2.03, 2.04, 2.04, 2.05, 2.06, 2.07, 2.08, 2.08, 2.08, 2.08, 2.09, 2.09, 2.10, 2.10, 2.11, 2.11,
      2.11, 2.12, 2.12, 2.13, 2.13, 2.14, 2.14, 2.14, 2.14, 2.15, 2.15, 2.16, 2.16, 2.16, 2.16, 2.16, 2.17};

  auto x = NDArrayFactory::create<double>(inputData, 'c', {1, 149});
  sd::ops::choose op;
  // greater than test
  auto result = op.evaluate({&x}, {0.0}, {3});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(1);

  ASSERT_EQ(148, z->e<double>(0));
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_ZERO) {
  std::vector<double> data;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
  }

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  sd::ops::choose op;
  // greater than test
  auto result = op.evaluate({&x}, {0.0}, {3});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(1);
  auto array = *z;
  ASSERT_EQ(3, array.e<double>(0));
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR) {
  std::vector<double> data;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
  }

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  auto scalar = NDArrayFactory::create<double>('c', {1, 1}, {0.0});
  sd::ops::choose op;
  // greater than test
  auto result = op.evaluate({&x, &scalar}, {1.0}, {3});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  ASSERT_EQ(3, z->lengthOf());
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_LEFT) {
  std::vector<double> data;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
  }

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  auto scalar = NDArrayFactory::create<double>('c', {1, 1}, {0.0});
  sd::ops::choose op;
  // greater than test
  auto result = op.evaluate({&scalar, &x}, {1.0}, {3});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  ASSERT_EQ(3, z->lengthOf());
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_ONLY_SCALAR) {
  std::vector<double> data;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
  }

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  sd::ops::choose op;
  // greater than test
  auto result = op.evaluate({&x}, {1.0}, {3});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  ASSERT_EQ(2, z->lengthOf());
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_ONLY_SCALAR_GTE) {
  std::vector<double> data;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
  }

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  sd::ops::choose op;
  // greater than test
  auto result = op.evaluate({&x}, {1.0}, {5});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  ASSERT_EQ(3, z->lengthOf());
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, TEST_WHERE) {
  std::vector<double> data;
  std::vector<bool> mask;
  std::vector<double> put;
  std::vector<double> resultData;
  std::vector<double> assertion;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
    if (i > 1) {
      assertion.push_back(5.0);
      mask.push_back(true);
    } else {
      assertion.push_back(i);
      mask.push_back(false);
    }

    put.push_back(5.0);
    resultData.push_back(0.0);
  }

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  auto maskArr = NDArrayFactory::create<bool>('c', {1, 4}, mask);
  auto putArr = NDArrayFactory::create<double>('c', {1, 4}, put);
  auto resultArr = NDArrayFactory::create<double>('c', {1, 4}, resultData);
  sd::ops::where_np op;
  // greater than test
  //            sd::Status execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs
  //            , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

  auto result = op.execute({&maskArr, &x, &putArr}, {&resultArr}, {}, {3}, {}, {}, false);
  ASSERT_EQ(sd::Status::OK, result);
  for (int i = 0; i < 4; i++) ASSERT_EQ(assertion[i], resultArr.e<double>(i));
  // auto z = result.at(0);
  // ASSERT_EQ(4,z->lengthOf());
  // ASSERT_TRUE(exp.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, TEST_WHERE_MASK) {
  double x[300] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double z[300] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool mask[300] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  double put[200] = {
      0.99666107, 0.9867112,  0.97686064, 0.9671082,  0.95745337, 0.9478948,  0.9384318,  0.92906314, 0.9197881,
      0.91060543, 0.9015147,  0.8925147,  0.8836044,  0.8747831,  0.86605,    0.85740393, 0.8488442,  0.84037,
      0.83198035, 0.8236745,  0.8154515,  0.8073106,  0.79925096, 0.79127187, 0.7833724,  0.77555174, 0.76780915,
      0.7601439,  0.75255525, 0.7450422,  0.7376043,  0.73024046, 0.72295034, 0.715733,   0.7085876,  0.7015135,
      0.69451016, 0.68757665, 0.6807124,  0.6739167,  0.66718876, 0.66052806, 0.6539338,  0.6474054,  0.6409421,
      0.6345435,  0.6282087,  0.6219371,  0.6157281,  0.60958105, 0.6034956,  0.59747064, 0.5915059,  0.5856007,
      0.57975453, 0.5739667,  0.5682366,  0.5625637,  0.5569475,  0.5513874,  0.54588276, 0.540433,   0.53503764,
      0.5296962,  0.52440816, 0.51917285, 0.5139898,  0.5088585,  0.50377846, 0.4987491,  0.4937699,  0.48884052,
      0.48396033, 0.47912875, 0.47434545, 0.4696099,  0.46492168, 0.46028027, 0.45568514, 0.4511359,  0.44663212,
      0.4421733,  0.43775895, 0.43338865, 0.42906195, 0.42477852, 0.4205379,  0.41633952, 0.41218308, 0.40806815,
      0.40399432, 0.3999611,  0.3959682,  0.39201516, 0.38810158, 0.384227,   0.38039115, 0.37659356, 0.37283397,
      0.3691119,  0.36542687, 0.36177874, 0.35816705, 0.3545914,  0.35105142, 0.34754673, 0.34407702, 0.34064204,
      0.33724132, 0.3338745,  0.33054137, 0.3272415,  0.32397458, 0.32074028, 0.3175382,  0.31436813, 0.31122974,
      0.3081226,  0.30504647, 0.30200112, 0.2989862,  0.29600134, 0.29304633, 0.2901207,  0.28722438, 0.28435695,
      0.2815181,  0.27870762, 0.27592525, 0.27317056, 0.27044344, 0.26774356, 0.26507056, 0.2624243,  0.25980446,
      0.25721073, 0.25464293, 0.25210077, 0.249584,   0.24709237, 0.24462552, 0.24218333, 0.23976555, 0.23737194,
      0.23500215, 0.23265606, 0.23033342, 0.22803394, 0.22575743, 0.2235036,  0.22127232, 0.21906327, 0.21687631,
      0.21471114, 0.21256764, 0.21044552, 0.20834461, 0.20626466, 0.20420544, 0.20216681, 0.20014854, 0.19815037,
      0.19617215, 0.19421372, 0.19227484, 0.19035533, 0.18845497, 0.18657354, 0.18471093, 0.18286693, 0.18104129,
      0.17923392, 0.17744459, 0.17567308, 0.1739193,  0.17218304, 0.17046405, 0.16876228, 0.16707748, 0.16540948,
      0.16375816, 0.16212334, 0.16050482, 0.15890247, 0.15731607, 0.15574552, 0.15419069, 0.15265137, 0.15112738,
      0.14961864, 0.14812498, 0.14664622, 0.1451822,  0.14373279, 0.14229788, 0.14087726, 0.13947085, 0.13807845,
      0.13669999, 0.13533528};
  double assertion[300] = {
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
      9.966611049434810354e-01, 9.867111603284486332e-01, 9.768605487739230320e-01, 9.671082786103732953e-01,
      9.574533680683808834e-01, 9.478948451798039354e-01, 9.384317476799283186e-01, 9.290631229105962285e-01,
      9.197880277243004610e-01, 9.106055283892373620e-01, 9.015147004953073528e-01, 8.925146288610534828e-01,
      8.836044074415293492e-01, 8.747831392370875037e-01, 8.660499362030764647e-01, 8.574039191604412302e-01,
      8.488442177072155204e-01, 8.403699701308978698e-01, 8.319803233217017979e-01, 8.236744326866727306e-01,
      8.154514620646623468e-01, 8.073105836421510251e-01, 7.992509778699116163e-01, 7.912718333805045523e-01,
      7.833723469065965173e-01, 7.755517232000953554e-01, 7.678091749520912224e-01, 7.601439227135980969e-01,
      7.525551948170853267e-01, 7.450422272987937689e-01, 7.376042638218265335e-01, 7.302405556000080011e-01,
      7.229503613225031211e-01, 7.157329470791886639e-01, 7.085875862867698771e-01, 7.015135596156351072e-01,
      6.945101549174396149e-01, 6.875766671534137009e-01, 6.807123983233853703e-01, 6.739166573955123196e-01,
      6.671887602367149173e-01, 6.605280295438040739e-01, 6.539337947752965619e-01, 6.474053920839111242e-01,
      6.409421642497381555e-01, 6.345434606140767375e-01, 6.282086370139332576e-01, 6.219370557171712832e-01,
      6.157280853583116942e-01, 6.095811008749726367e-01, 6.034954834449430816e-01, 5.974706204238864338e-01,
      5.915059052836644238e-01, 5.856007375512777280e-01, 5.797545227484157682e-01, 5.739666723316099173e-01,
      5.682366036329845604e-01, 5.625637398015992385e-01, 5.569475097453767676e-01, 5.513873480736106725e-01,
      5.458826950400470501e-01, 5.404329964865340896e-01, 5.350377037872348085e-01, 5.296962737933965659e-01,
      5.244081687786711354e-01, 5.191728563849821176e-01, 5.139898095689314772e-01, 5.088585065487419845e-01,
      5.037784307517284565e-01, 4.987490707622945774e-01, 4.937699202704479151e-01, 4.888404780208293054e-01,
      4.839602477622509946e-01, 4.791287381977387683e-01, 4.743454629350723484e-01, 4.696099404378203390e-01,
      4.649216939768630041e-01, 4.602802515824001017e-01, 4.556851459964368911e-01, 4.511359146257447605e-01,
      4.466320994952920342e-01, 4.421732472021388527e-01, 4.377589088697927955e-01, 4.333886401030203062e-01,
      4.290620009431086457e-01, 4.247785558235752101e-01, 4.205378735263185508e-01, 4.163395271382073215e-01,
      4.121830940081024908e-01, 4.080681557043087104e-01, 4.039942979724505667e-01, 3.999611106937689398e-01,
      3.959681878438343627e-01, 3.920151274516718853e-01, 3.881015315592946102e-01, 3.842270061816405180e-01,
      3.803911612669100828e-01, 3.765936106572991271e-01, 3.728339720501240850e-01, 3.691118669593352886e-01,
      3.654269206774144463e-01, 3.617787622376523182e-01, 3.581670243768036999e-01, 3.545913434981138868e-01,
      3.510513596347161203e-01, 3.475467164133922426e-01, 3.440770610186974499e-01, 3.406420441574410929e-01,
      3.372413200235238606e-01, 3.338745462631242389e-01, 3.305413839402346898e-01, 3.272414975025391692e-01,
      3.239745547476344245e-01, 3.207402267895853032e-01, 3.175381880258169032e-01, 3.143681161043347383e-01,
      3.112296918912743071e-01, 3.081225994387726264e-01, 3.050465259531625062e-01, 3.020011617634821843e-01,
      2.989862002903017069e-01, 2.960013380148582840e-01, 2.930462744485015647e-01, 2.901207121024425017e-01,
      2.872243564578055852e-01, 2.843569159359789489e-01, 2.815181018692606840e-01, 2.787076284717992514e-01,
      2.759252128108221624e-01, 2.731705747781537075e-01, 2.704434370620155681e-01, 2.677435251191103149e-01,
      2.650705671469821278e-01, 2.624242940566549609e-01, 2.598044394455423789e-01, 2.572107395706292876e-01,
      2.546429333219200064e-01, 2.521007621961529055e-01, 2.495839702707757235e-01, 2.470923041781825646e-01,
      2.446255130802063582e-01, 2.421833486428674187e-01, 2.397655650113727777e-01, 2.373719187853666479e-01,
      2.350021689944260528e-01, 2.326560770738031469e-01, 2.303334068404078172e-01, 2.280339244690317291e-01,
      2.257573984688081292e-01, 2.235035996599082919e-01, 2.212723011504689752e-01, 2.190632783137518302e-01,
      2.168763087655291855e-01, 2.147111723416972873e-01, 2.125676510761114746e-01, 2.104455291786438698e-01,
      2.083445930134591173e-01, 2.062646310775079761e-01, 2.042054339792348794e-01, 2.021667944174980747e-01,
      2.001485071607009836e-01, 1.981503690261307848e-01, 1.961721788595043592e-01, 1.942137375147174327e-01,
      1.922748478337968081e-01, 1.903553146270518526e-01, 1.884549446534251604e-01, 1.865735466010380594e-01,
      1.847109310679319050e-01, 1.828669105430000552e-01, 1.810412993871116094e-01, 1.792339138144224131e-01,
      1.774445718738737465e-01, 1.756730934308744496e-01, 1.739193001491673995e-01, 1.721830154728755669e-01,
      1.704640646087285105e-01, 1.687622745084652875e-01, 1.670774738514141378e-01, 1.654094930272448083e-01,
      1.637581641188943782e-01, 1.621233208856623365e-01, 1.605047987464754966e-01, 1.589024347633189727e-01,
      1.573160676248336609e-01, 1.557455376300762306e-01, 1.541906866724424563e-01, 1.526513582237501165e-01,
      1.511273973184814046e-01, 1.496186505381822129e-01, 1.481249659960175158e-01, 1.466461933214808777e-01,
      1.451821836452561187e-01, 1.437327895842310799e-01, 1.422978652266598532e-01, 1.408772661174743090e-01,
      1.394708492437411185e-01, 1.380784730202649913e-01, 1.366999972753347725e-01, 1.353352832366127023e-01};
  sd::LongType threeHundredShapePointer[8] = {2, 1, 300, 1, 1, 0, 1, 99};
  sd::LongType twoHundredShapePointer[8] = {2, 1, 200, 1, 1, 0, 1, 99};
  sd::ops::where_np op;
  ArrayOptions::setDataType(threeHundredShapePointer, sd::DataType::DOUBLE);
  ArrayOptions::setDataType(twoHundredShapePointer, sd::DataType::DOUBLE);

  NDArray xArr(x, threeHundredShapePointer);
  NDArray putArr(put, twoHundredShapePointer);
  NDArray resultArr(z, threeHundredShapePointer);

  resultArr.assign(0.0);
  ArrayOptions::setDataType(threeHundredShapePointer, sd::DataType::BOOL);
  NDArray maskArr(mask, threeHundredShapePointer);

  ArrayOptions::setDataType(threeHundredShapePointer, sd::DataType::DOUBLE);
  NDArray assertArr(assertion, threeHundredShapePointer);
  sd::Status result = op.execute({&maskArr, &xArr, &putArr}, {&resultArr}, {}, {}, {});
  ASSERT_EQ(sd::Status::OK, result);
  ASSERT_TRUE(assertArr.isSameShape(resultArr));
  ASSERT_TRUE(assertArr.equalsTo(resultArr));
}

TEST_F(DeclarableOpsTests7, TEST_WHERE_SCALAR) {
  std::vector<double> data;
  std::vector<bool> mask;
  std::vector<double> put;
  std::vector<double> resultData;
  std::vector<double> assertion;
  for (sd::LongType i = 0; i < 4; i++) {
    data.push_back(i);
    if (i > 1) {
      assertion.push_back(5.0);
      mask.push_back(true);
    } else {
      assertion.push_back(i);
      mask.push_back(false);
    }

    resultData.push_back(0.0);
  }

  put.push_back(5.0);

  auto x = NDArrayFactory::create<double>('c', {1, 4}, data);
  auto maskArr = NDArrayFactory::create<bool>('c', {1, 4}, mask);
  auto putArr = NDArrayFactory::create<double>('c', {1, 1}, put);
  auto resultArr = NDArrayFactory::create<double>('c', {1, 4}, resultData);
  sd::ops::where_np op;
  // greater than test
  //            sd::Status execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs
  //            , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

  auto result = op.execute({&maskArr, &x, &putArr}, {&resultArr}, {}, {3}, {}, {}, false);
  // ASSERT_EQ(sd::Status::OK, result.status());
  for (int i = 0; i < 4; i++) ASSERT_EQ(assertion[i], resultArr.e<double>(i));
  // auto z = result.at(0);
  // ASSERT_EQ(4,z->lengthOf());
  // ASSERT_TRUE(exp.isSameShape(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiagPart_1) {
  auto x =
      NDArrayFactory::create<double>('c', {2, 4, 4}, {1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.,
                                                      5., 0., 0., 0., 0., 6., 0., 0., 0., 0., 7., 0., 0., 0., 0., 8.});

  auto z = NDArrayFactory::create<double>('c', {2, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

  sd::ops::matrix_diag_part op;

  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(z.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiagPart_2) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 3, 4}, {1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 5., 0., 0., 0., 0., 6., 0., 0., 0., 0., 7., 0.});

  auto z = NDArrayFactory::create<double>('c', {2, 3}, {1, 2, 3, 5, 6, 7});

  sd::ops::matrix_diag_part op;

  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(z.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiag_1) {
  auto z =
      NDArrayFactory::create<double>('c', {2, 4, 4}, {1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.,
                                                      5., 0., 0., 0., 0., 6., 0., 0., 0., 0., 7., 0., 0., 0., 0., 8.});

  auto x = NDArrayFactory::create<double>('c', {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});

  sd::ops::matrix_diag op;

  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(z.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestMatrixDiag_2) {
  auto z = NDArrayFactory::create<double>('c', {2, 3, 3},
                                          {1., 0., 0., 0., 2., 0., 0., 0., 3., 5., 0., 0., 0., 6., 0., 0., 0., 7.});
  auto x = NDArrayFactory::create<double>('c', {2, 3}, {1, 2, 3, 5, 6, 7});

  sd::ops::matrix_diag op;

  auto result = op.evaluate({&x}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(z.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRandomCrop_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 2, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto shape = NDArrayFactory::create<int>({1, 2, 3});
  sd::ops::random_crop op;

  auto result = op.evaluate({&x, &shape}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRandomCrop_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 2, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto shape = NDArrayFactory::create<sd::LongType>({2, 2, 2});
  sd::ops::random_crop op;

  auto result = op.evaluate({&x, &shape}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  //    ASSERT_TRUE(z.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Dynamic_Stitch_119) {
  auto indices0 = NDArrayFactory::create<int>('c', {2}, {1, 10});
  auto indices1 = NDArrayFactory::create<int>('c', {2, 3}, {0, 7, 9, 5, 8, 3});
  auto indices2 = NDArrayFactory::create<int>('c', {3, 1}, {6, 4, 2});
  auto data0 = NDArrayFactory::create<double>(
      'c', {2, 5, 4}, {1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f, 13.f, 14.f,
                       15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f,
                       29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f});

  auto data1 = NDArrayFactory::create<double>(
      'c', {2, 3, 5, 4},
      {1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,   10.f,  11.f,  12.f,  13.f,  14.f,  15.f,
       16.f,  17.f,  18.f,  19.f,  20.f,  21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,  29.f,  30.f,
       31.f,  32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f,  41.f,  42.f,  43.f,  44.f,  45.f,
       46.f,  47.f,  48.f,  49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,
       61.f,  62.f,  63.f,  64.f,  65.f,  66.f,  67.f,  68.f,  69.f,  70.f,  71.f,  72.f,  73.f,  74.f,  75.f,
       76.f,  77.f,  78.f,  79.f,  80.f,  81.f,  82.f,  83.f,  84.f,  85.f,  86.f,  87.f,  88.f,  89.f,  90.f,
       91.f,  92.f,  93.f,  94.f,  95.f,  96.f,  97.f,  98.f,  99.f,  100.f, 101.f, 102.f, 103.f, 104.f, 105.f,
       106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f, 120.f});

  auto data2 = NDArrayFactory::create<double>(
      'c', {3, 1, 5, 4}, {1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
                          16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f,
                          31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f,
                          46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f});

  auto exp = NDArrayFactory::create<double>(
      'c', {11, 5, 4},
      {1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,   10.f,  11.f,  12.f,  13.f,  14.f,  15.f,  16.f,
       17.f,  18.f,  19.f,  20.f,  1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,   10.f,  11.f,  12.f,
       13.f,  14.f,  15.f,  16.f,  17.f,  18.f,  19.f,  20.f,  41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,  48.f,
       49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,  101.f, 102.f, 103.f, 104.f,
       105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f, 120.f,
       21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,  29.f,  30.f,  31.f,  32.f,  33.f,  34.f,  35.f,  36.f,
       37.f,  38.f,  39.f,  40.f,  61.f,  62.f,  63.f,  64.f,  65.f,  66.f,  67.f,  68.f,  69.f,  70.f,  71.f,  72.f,
       73.f,  74.f,  75.f,  76.f,  77.f,  78.f,  79.f,  80.f,  1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,
       9.f,   10.f,  11.f,  12.f,  13.f,  14.f,  15.f,  16.f,  17.f,  18.f,  19.f,  20.f,  21.f,  22.f,  23.f,  24.f,
       25.f,  26.f,  27.f,  28.f,  29.f,  30.f,  31.f,  32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f,
       81.f,  82.f,  83.f,  84.f,  85.f,  86.f,  87.f,  88.f,  89.f,  90.f,  91.f,  92.f,  93.f,  94.f,  95.f,  96.f,
       97.f,  98.f,  99.f,  100.f, 41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,  48.f,  49.f,  50.f,  51.f,  52.f,
       53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,  21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,
       29.f,  30.f,  31.f,  32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f});

  sd::ops::dynamic_stitch op;
  auto result = op.evaluate({&indices0, &indices1, &indices2, &data0, &data1, &data2}, {}, {});
  ASSERT_EQ(sd::Status::OK, result.status());
  //    result.at(0)->printIndexedBuffer("Output");
  //    exp.printIndexedBuffer("Expect");
  //    result.at(0)->printShapeInfo("Output shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Dynamic_Stitch_Prof_1) {
  auto indices0 = NDArrayFactory::create<int>('c', {2}, {1, 10});
  auto indices1 = NDArrayFactory::create<int>('c', {2, 3}, {0, 7, 9, 5, 8, 3});
  auto indices2 = NDArrayFactory::create<int>('c', {3, 1}, {6, 4, 2});
  auto data0 = NDArrayFactory::create<double>(
      'c', {2, 5, 4}, {1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f, 13.f, 14.f,
                       15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f,
                       29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f});

  auto data1 = NDArrayFactory::create<double>(
      'c', {2, 3, 5, 4},
      {1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,   10.f,  11.f,  12.f,  13.f,  14.f,  15.f,
       16.f,  17.f,  18.f,  19.f,  20.f,  21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,  29.f,  30.f,
       31.f,  32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f,  41.f,  42.f,  43.f,  44.f,  45.f,
       46.f,  47.f,  48.f,  49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,
       61.f,  62.f,  63.f,  64.f,  65.f,  66.f,  67.f,  68.f,  69.f,  70.f,  71.f,  72.f,  73.f,  74.f,  75.f,
       76.f,  77.f,  78.f,  79.f,  80.f,  81.f,  82.f,  83.f,  84.f,  85.f,  86.f,  87.f,  88.f,  89.f,  90.f,
       91.f,  92.f,  93.f,  94.f,  95.f,  96.f,  97.f,  98.f,  99.f,  100.f, 101.f, 102.f, 103.f, 104.f, 105.f,
       106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f, 120.f});

  auto data2 = NDArrayFactory::create<double>(
      'c', {3, 1, 5, 4}, {1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
                          16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f,
                          31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f,
                          46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f});

  auto exp = NDArrayFactory::create<double>(
      'c', {11, 5, 4},
      {1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,   10.f,  11.f,  12.f,  13.f,  14.f,  15.f,  16.f,
       17.f,  18.f,  19.f,  20.f,  1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,   10.f,  11.f,  12.f,
       13.f,  14.f,  15.f,  16.f,  17.f,  18.f,  19.f,  20.f,  41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,  48.f,
       49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,  101.f, 102.f, 103.f, 104.f,
       105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f, 120.f,
       21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,  29.f,  30.f,  31.f,  32.f,  33.f,  34.f,  35.f,  36.f,
       37.f,  38.f,  39.f,  40.f,  61.f,  62.f,  63.f,  64.f,  65.f,  66.f,  67.f,  68.f,  69.f,  70.f,  71.f,  72.f,
       73.f,  74.f,  75.f,  76.f,  77.f,  78.f,  79.f,  80.f,  1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,
       9.f,   10.f,  11.f,  12.f,  13.f,  14.f,  15.f,  16.f,  17.f,  18.f,  19.f,  20.f,  21.f,  22.f,  23.f,  24.f,
       25.f,  26.f,  27.f,  28.f,  29.f,  30.f,  31.f,  32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f,
       81.f,  82.f,  83.f,  84.f,  85.f,  86.f,  87.f,  88.f,  89.f,  90.f,  91.f,  92.f,  93.f,  94.f,  95.f,  96.f,
       97.f,  98.f,  99.f,  100.f, 41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,  48.f,  49.f,  50.f,  51.f,  52.f,
       53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,  21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,
       29.f,  30.f,  31.f,  32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f});

  sd::ops::dynamic_stitch op;
  auto result = op.evaluate({&indices0, &indices1, &indices2, &data0, &data1, &data2}, {}, {});
  ASSERT_EQ(sd::Status::OK, result.status());
  //    result.at(0)->printIndexedBuffer("Output");
  //    exp.printIndexedBuffer("Expect");
  //    result.at(0)->printShapeInfo("Output shape");
  auto res = result.at(0);
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
  int numOfCases = 100;
  auto timeStart = std::chrono::system_clock::now();

  for (int i = 0; i < numOfCases; i++) {
    op.execute({&indices0, &indices1, &indices2, &data0, &data1, &data2}, {res}, {}, {}, {});
  }

  auto timeEnd = std::chrono::system_clock::now();
  auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
  // sd_printf("dynamic_stitch: Process with %i iterations was load: %lld us.\n", numOfCases, outerTime / numOfCases);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Dynamic_Stitch_119_1) {
  auto indices0 = NDArrayFactory::create<int>('c', {2}, {1, 10});
  auto indices1 = NDArrayFactory::create<int>('c', {2, 3}, {0, 7, 9, 5, 8, 3});
  auto indices2 = NDArrayFactory::create<int>('c', {3, 1}, {6, 4, 2});

  auto data0 = NDArrayFactory::create<double>('c', {2, 5, 4});
  auto data1 = NDArrayFactory::create<double>('c', {2, 3, 5, 4});
  auto data2 = NDArrayFactory::create<double>('c', {3, 1, 5, 4});

  auto exp = NDArrayFactory::create<double>(
      'c', {11, 5, 4},
      {
          21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,

          1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,

          181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,

          121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,

          161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,

          81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100,

          141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,

          41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,

          101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

          61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,

          21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
      });
  data0.linspace(1);
  data1.linspace(21);
  data2.linspace(141);
  sd::ops::dynamic_stitch op;
  auto result = op.evaluate({&indices0, &indices1, &indices2, &data0, &data1, &data2}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);

  ASSERT_TRUE(z->isSameShape(exp));
  ASSERT_TRUE(z->equalsTo(exp));
}

TEST_F(DeclarableOpsTests7, Test_Dynamic_Stitch_119_2) {
  auto indices0 = NDArrayFactory::create<int>('c', {2}, {1, 10});
  auto indices1 = NDArrayFactory::create<int>('c', {2, 3}, {0, 7, 9, 5, 8, 3});
  auto indices2 = NDArrayFactory::create<int>('c', {3, 1}, {6, 4, 2});

  auto data0 = NDArrayFactory::create<double>('c', {2, 5, 4});
  auto data1 = NDArrayFactory::create<double>('c', {2, 3, 5, 4});
  auto data2 = NDArrayFactory::create<double>('c', {3, 1, 5, 4});

  auto exp = NDArrayFactory::create<double>(
      'c', {11, 5, 4},
      {
          41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,

          1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,

          201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,

          141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,

          181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,

          101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

          161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,

          61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,

          121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,

          81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100,

          21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
      });
  data0.linspace(1);
  data1.linspace(41);
  data2.linspace(161);
  sd::ops::dynamic_stitch op;
  auto result = op.evaluate({&indices0, &indices1, &indices2, &data0, &data1, &data2}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);

  ASSERT_TRUE(z->isSameShape(exp));
  ASSERT_TRUE(z->equalsTo(exp));
}

TEST_F(DeclarableOpsTests7, Test_Dynamic_Partition_119) {
  auto x = NDArrayFactory::create<double>('c', {5, 4, 11});
  auto y = NDArrayFactory::create<double>('c', {5, 4}, {0, 1, 2, 3, 1, 0, 2, 3, 2, 3, 1, 0, 2, 1, 0, 3, 0, 1, 2, 3});
  auto e = NDArrayFactory::create<double>('c', {5, 11});
  x.assign(1.f);
  e.assign(1.f);
  sd::ops::dynamic_partition op;
  auto result = op.evaluate({&x, &y}, {}, {4});
  ASSERT_EQ(sd::Status::OK, result.status());
  ASSERT_EQ(4, result.size());
  auto z = result.at(0);

  ASSERT_TRUE(e.isSameShape(z));
}

TEST_F(DeclarableOpsTests7, Test_Dynamic_Partition_119_1) {
  auto x = NDArrayFactory::create<double>(
      'c', {3, 4, 2}, {10, 20, 11, 21, 12, 22, 13, 23, 14, 24, 15, 25, 16, 26, 17, 27, 18, 28, 19, 29, 20, 30, 21, 31});

  auto y = NDArrayFactory::create<int>('c', {3, 4}, {0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1});
  auto e = NDArrayFactory::create<double>('c', {4, 2}, {10, 20, 11, 21, 12, 22, 13, 23});

  //    x.assign(1.f);
  //    e.assign(1.f);
  sd::ops::dynamic_partition op;
  auto result = op.evaluate({&x, &y}, {}, {3});
  ASSERT_EQ(sd::Status::OK, result.status());
  ASSERT_EQ(3, result.size());
  auto z = result.at(0);
  //    z->printShapeInfo("Output shape info");
  //    result.at(1)->printShapeInfo("Shape2");
  //    result.at(2)->printShapeInfo("Shape3");
  //    result.at(3)->printShapeInfo("Shape4");
  //    z->printIndexedBuffer("Output1");
  //    result.at(1)->printIndexedBuffer("Output2");
  //    result.at(2)->printIndexedBuffer("Output3");
  //    result.at(3)->printIndexedBuffer("Output4");
  ASSERT_TRUE(e.isSameShape(z));
}
TEST_F(DeclarableOpsTests7, Test_Dynamic_Partition_119_2) {
  auto x = NDArrayFactory::create<double>('c', {5, 4, 11});
  auto y = NDArrayFactory::create<int>('c', {5, 4}, {0, 1, 2, 3, 1, 0, 2, 3, 2, 3, 1, 0, 2, 1, 0, 3, 0, 1, 2, 3});
  auto e1 = NDArrayFactory::create<double>(
      'c', {5, 11}, {1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  56,  57,  58,  59,  60,  61,  62,  63,
                     64,  65,  66,  122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 155, 156, 157, 158, 159,
                     160, 161, 162, 163, 164, 165, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187});
  auto e2 = NDArrayFactory::create<double>(
      'c', {5, 11}, {12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  45,  46,  47,  48,  49,  50,  51,  52,
                     53,  54,  55,  111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 144, 145, 146, 147, 148,
                     149, 150, 151, 152, 153, 154, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198});
  auto e3 = NDArrayFactory::create<double>(
      'c', {5, 11}, {23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  67,  68,  69,  70,  71,  72,  73,  74,
                     75,  76,  77,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  133, 134, 135, 136, 137,
                     138, 139, 140, 141, 142, 143, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209});
  auto e4 = NDArrayFactory::create<double>(
      'c', {5, 11}, {34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  78,  79,  80,  81,  82,  83,  84,  85,
                     86,  87,  88,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 166, 167, 168, 169, 170,
                     171, 172, 173, 174, 175, 176, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220});
  std::vector<NDArray*> e({&e1, &e2, &e3, &e4});
  x.linspace(1.f);
  //.assign(1.f);
  sd::ops::dynamic_partition op;
  auto result = op.evaluate({&x, &y}, {}, {4});
  ASSERT_EQ(sd::Status::OK, result.status());
  ASSERT_EQ(4, result.size());
  for (size_t i = 0; i < result.size(); i++) {
    auto z = result.at(i);
    //    z->printShapeInfo("Output shape info");
    //     z->printIndexedBuffer("Output1");
    //    result.at(1)->printIndexedBuffer("Output2");
    //    result.at(2)->printIndexedBuffer("Output3");
    //    result.at(3)->printIndexedBuffer("Output4");
    ASSERT_TRUE(e[i]->isSameShape(z));
    ASSERT_TRUE(e[i]->equalsTo(z));
  }
}

TEST_F(DeclarableOpsTests7, Test_SequenceMask_1) {
  auto input = NDArrayFactory::create<int>('c', {4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  auto exp = NDArrayFactory::create<bool>(
      'c', {4, 4, 16},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  sd::ops::sequence_mask op;
  auto result = op.evaluate({&input}, {}, {});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("Output");
  //    z->printShapeInfo("Shape");
  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests7, Test_SequenceMask_2) {
  auto input = NDArrayFactory::create<int>('c', {2, 2, 2}, {10, 20, 30, 4, 0, 6, 7, 8});
  auto exp = NDArrayFactory::create<bool>(
      'c', {2, 2, 2, 30},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  sd::ops::sequence_mask op;
  auto result = op.evaluate({&input}, {}, {});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printBuffer("Output");
  //    z->printShapeInfo("Shape");
  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests7, Test_SequenceMask_3) {
  auto input = NDArrayFactory::create<int>('c', {2, 2, 2}, {10, 20, 30, 4, 0, 6, 7, 8});
  auto exp = NDArrayFactory::create<int>(
      'c', {2, 2, 2, 30},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  sd::ops::sequence_mask op;
  auto result = op.evaluate({&input}, {sd::DataType::INT32});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printBuffer("Output");
  //    z->printShapeInfo("Shape");
  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests7, Test_SequenceMask_4) {
  auto input = NDArrayFactory::create<int>({1, 3, 2});
  auto maxLen = NDArrayFactory::create<int>(5);
  auto exp = NDArrayFactory::create<float>('c', {3, 5},
                                           {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f});

  sd::ops::sequence_mask op;
  auto result = op.evaluate({&input, &maxLen}, {sd::DataType::FLOAT32});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printBuffer("Output");
  //    z->printShapeInfo("Shape");
  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(DeclarableOpsTests7, Test_SequenceMask_5) {
  auto input = NDArrayFactory::create<int>({1, 3, 2});
  auto exp = NDArrayFactory::create<float>('c', {3, 5},
                                           {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f});

  sd::ops::sequence_mask op;
  auto result = op.evaluate({&input}, {5, (int)sd::DataType::FLOAT32});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printBuffer("Output");
  //    z->printShapeInfo("Shape");
  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({2.5, 9, 3, 9, 4.2});

  sd::ops::segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printBuffer("MaX1");
  //    exp.printBuffer("ExP1");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_01) {
  auto x = NDArrayFactory::create<double>(
      {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1., 10, 40, 30});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5});
  auto exp = NDArrayFactory::create<double>({2.5, 9, 3, 9, 4.2, 40});

  sd::ops::segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printBuffer("MaX01");
  //    exp.printBuffer("ExP01");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMaxBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({0., 1., 0., 2., 0., 0., 3., 4., 0., 0., 0., 0., 0., 5., 0., 0.});
  auto eps = NDArrayFactory::create<double>('c', {5});
  sd::ops::segment_max_bp op;
  eps.linspace(1);
  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("OutputMaxBP");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_2) {
  auto x = NDArrayFactory::create<double>(
      'c', {5, 4}, {0, 1.8, 2.5, 4., 1, 9., 2.1, 2.4, 0, 3., 9., 2.1, 2, 1, 2.1, 0.7, 3, 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1, 9, 9, 4, 2, 1, 2.1, 0.7, 3, 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}
  sd::ops::segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  auto out = result.at(0);
  //    out->printIndexedBuffer("Output2Max");
  //    exp.printIndexedBuffer("Expect2Max");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMaxBP_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto eps = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  //    NDArray<double> exp('c', {3, 4}, {2.1, 2.5, 4, 9,2.1, 2.1, 0.7, 0.1,3., 4.2, 2.2, 1.});
  auto exp =
      NDArrayFactory::create<double>('c', {4, 4}, {0., 2., 3., 4., 1., 0., 0., 4., 5., 6., 7., 8., 9., 10., 11., 12.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::segment_max_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  // exp.printIndexedBuffer("BP Max Expect");
  // result.at(0)->printIndexedBuffer("BP Max Output");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 87., 44., 55.1, 56.4, 93., 28., 119.1, 82.1, 112.7, 113.1, 114., 114.2, 116.2, 117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output3Max");
  //    result.at(0)->printShapeInfo("Out Shape 3 Max");
  //    exp.printIndexedBuffer("Expect3Max");
  //    exp.printShapeInfo("Exp Shape 3 Max");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMax_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMax_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({4, 4, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0});
  auto exp = NDArrayFactory::create<double>({2.2, 9., 3., 9., 4.2});

  sd::ops::unsorted_segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMaxBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({0., 1., 0., 2., 0., 0., 3., 4., 0., 0., 0., 0., 0., 5., 0., 0.});
  auto eps = NDArrayFactory::create<double>('c', {5});
  sd::ops::segment_max_bp op;
  eps.linspace(1);
  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMaxBP_2) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({3., 0., 1., 0., 2., 0., 0., 4., 0., 0., 0., 0., 0., 5., 0., 0.});
  auto eps = NDArrayFactory::create<double>('c', {5});
  sd::ops::segment_max_bp op;
  eps.linspace(1);
  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMax_2) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({4, 4, 1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0});
  auto exp = NDArrayFactory::create<double>({2.2, 9., -DataTypeUtils::max<double>(), 9., 4.2});

  sd::ops::unsorted_segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("OutputUnsortedMax");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMax_3) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {2.1, 2.5, 4, 9, 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::unsorted_segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  // exp.printIndexedBuffer("Expect");
  // result.at(0)->printIndexedBuffer("Output");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMax_4) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 8., 2.1, 2.1, 11.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 0, 2});
  double principalMax = DataTypeUtils::max<double>();
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4}, {2.1, 2.5, 11.7, 9, -principalMax, -principalMax, -principalMax, -principalMax, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::unsorted_segment_max op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  // exp.printIndexedBuffer("Expect");
  // result.at(0)->printIndexedBuffer("Output");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMin_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({1.8, 2.1, 3., 2.1, 0.1});

  sd::ops::segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMin_01) {
  auto x =
      NDArrayFactory::create<double>({1.8, -2.5, 4., -9., 2.1, 2.4, -3., -9., 2.1, 2.1, 0.7, 0.1, 3., -4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({-2.5, -9, -3., -9, -4.2});

  sd::ops::segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMin_02) {
  auto x = NDArrayFactory::create<float>(
      {1.8f, -2.5f, 4.f, -9.f, 2.1f, 2.4f, -3.f, -9.f, 2.1f, 2.1f, 0.7f, 0.1f, 3.f, -4.2f, 2.2f, 1.f});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<float>({-2.5f, -9.f, -3.f, -9.f, -4.2f});

  sd::ops::segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMinBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({1., 0., 0., 0., 2., 0., 3., 0., 4., 4., 0., 5., 0., 0., 0., 0.});
  auto eps = NDArrayFactory::create<double>('c', {5});
  eps.linspace(1);
  sd::ops::segment_min_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMinBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({1., 0., 0., 0., 2., 0., 3., 0., 4., 4., 0., 5., 0., 0., 0., 0.});
  auto eps = NDArrayFactory::create<double>('c', {5});
  eps.linspace(1);
  sd::ops::unsorted_segment_min_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output1");
  // exp.printIndexedBuffer("Expecte");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMinBP_2) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({3., 1., 0., 0., 0., 2., 0., 0., 4., 4., 0., 5., 0., 0., 0., 0.});
  auto eps = NDArrayFactory::create<double>('c', {5});
  eps.linspace(1);
  sd::ops::unsorted_segment_min_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output1");
  // exp.printIndexedBuffer("Expecte");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMin_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.8, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMinBP_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<sd::LongType>({0, 0, 1, 2});
  auto eps = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  auto exp =
      NDArrayFactory::create<double>('c', {4, 4}, {1., 0., 0., 4., 0., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::segment_min_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  //    exp.printIndexedBuffer("Expect");
  //    result.at(0)->printIndexedBuffer("Output");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMin_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.,
                       31., 22., 67., 24., 15.1, 46.4, 73., 28., 109.1, 12.1, 12.7,  13.1, 14., 14.2,  16.2, 11.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.});

  sd::ops::segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMin_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMin_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({1.8, 2.1, 3., 2.1, 0.1});

  sd::ops::unsorted_segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMin_01) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({1.8, 2.1, 3., 2.1, 0.1});

  sd::ops::unsorted_segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMin_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.8, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::unsorted_segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMin_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.,
                       31., 22., 67., 24., 15.1, 46.4, 73., 28., 109.1, 12.1, 12.7,  13.1, 14., 14.2,  16.2, 11.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.});

  sd::ops::unsorted_segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMin_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  double principalMax = DataTypeUtils::max<double>();

  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4},
      {91.,          82.,          37.,          64.,          55.1,         46.4,         73.,          28.,
       119.1,        12.1,         112.7,        13.1,         14.,          114.2,        16.2,         117.,
       51.,          42.,          67.,          24.,          15.1,         56.4,         93.,          28.,
       109.1,        82.1,         12.7,         113.1,        114.,         14.2,         116.2,        11.,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       31.,          22.,          87.,          44.,          55.1,         46.4,         73.,          28.,
       119.1,        12.1,         112.7,        13.1,         14.,          114.2,        16.2,         117.,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax, principalMax,
       91.,          82.,          37.,          64.,          55.1,         46.4,         73.,          28.,
       119.1,        12.1,         112.7,        13.1,         14.,          114.2,        16.2,         117.});

  sd::ops::unsorted_segment_min op;

  auto result = op.evaluate({&x, &idx}, {}, {8});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMean_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({2.15, 4.375, 3., 4.4, 1.8666667});

  sd::ops::segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

TEST_F(DeclarableOpsTests7, TestSegmentMean_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.95, 2.45, 3.5, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  sd::ops::segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    result.at(0)->printIndexedBuffer("Output");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

TEST_F(DeclarableOpsTests7, TestSegmentMean_02) {
  auto x = NDArrayFactory::create<double>(
      'c', {6, 3}, {1, 2, 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 2, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 3}, {2.5, 3.5, 4.5, 8.5, 9.5, 10.5, 14.5, 15.5, 16.5});

  sd::ops::segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

TEST_F(DeclarableOpsTests7, TestSegmentMean_021) {
  auto x = NDArrayFactory::create<float>(
      'c', {6, 3});  //, {1, 2,  3., 4., 5., 6.,  7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 2, 2});
  auto exp = NDArrayFactory::create<float>('c', {3, 3}, {2.5f, 3.5f, 4.5f, 8.5f, 9.5f, 10.5f, 14.5f, 15.5f, 16.5f});

  sd::ops::segment_mean op;
  x.linspace(1.);
  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

TEST_F(DeclarableOpsTests7, TestSegmentMean_022) {
  auto x = NDArrayFactory::create<float>(
      'c', {6, 3});  //, {1, 2,  3., 4., 5., 6.,  7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 2, 2});
  auto z =
      NDArrayFactory::create<float>('c', {3, 3});  //, {    2.5, 3.5, 4.5,      8.5, 9.5, 10.5,   14.5, 15.5,  16.5});
  auto exp = NDArrayFactory::create<float>('c', {3, 3}, {2.5f, 3.5f, 4.5f, 8.5f, 9.5f, 10.5f, 14.5f, 15.5f, 16.5f});

  sd::ops::segment_mean op;
  x.linspace(1.);
  auto result = op.execute({&x, &idx}, {&z});
  ASSERT_EQ(result, sd::Status::OK);

  ASSERT_TRUE(exp.equalsTo(z));

  //
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMeanBP_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto eps = NDArrayFactory::create<double>('c', {3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4, 4},
                                            {0.5, 1., 1.5, 2., 0.5, 1., 1.5, 2., 5., 6., 7., 8., 9., 10., 11., 12.});
  eps.linspace(1);

  sd::ops::segment_mean_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMean_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.,
                       41., 32., 77., 34., 35.1, 51.4, 83., 28., 114.1, 47.1, 62.7,  63.1, 64., 64.2,  66.2, 64.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.});

  sd::ops::segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMean_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMean_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({2.15, 4.375, 3., 4.4, 1.8666667});

  sd::ops::unsorted_segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentMeanBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>({1. / 2., 1. / 2., 2. / 4., 2. / 4., 2. / 4., 2. / 4, 3., 4. / 3., 4. / 3.,
                                             4. / 3., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6.});
  sd::ops::segment_mean_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMeanBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>({1. / 2., 1. / 2., 2. / 4., 2. / 4., 2. / 4., 2. / 4, 3., 4. / 3., 4. / 3.,
                                             4. / 3., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6.});
  sd::ops::unsorted_segment_mean_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMeanBP_2) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>({3., 1. / 2., 1. / 2., 2. / 4., 2. / 4., 2. / 4., 2. / 4, 4. / 3., 4. / 3.,
                                             4. / 3., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6.});
  sd::ops::unsorted_segment_mean_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMean_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.95, 2.45, 3.5, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  sd::ops::unsorted_segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    result.at(0)->printIndexedBuffer("Output");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMean_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.,
                       41., 32., 77., 34., 35.1, 51.4, 83., 28., 114.1, 47.1, 62.7,  63.1, 64., 64.2,  66.2, 64.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1, 14., 114.2, 16.2, 117.});

  sd::ops::unsorted_segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentMean_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::unsorted_segment_mean op;

  auto result = op.evaluate({&x, &idx}, {}, {8});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({3.0405593, 8.75, 3., 7.621024, 4.5723805});

  sd::ops::unsorted_segment_sqrt_n op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_BP_1) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  //    NDArray<double> exp({3.0405593, 8.75,      3.,        7.621024,  4.5723805});
  auto exp = NDArrayFactory::create<double>({3., 0.707107, 0.707107, 1., 1., 1., 1., 2.309401, 2.309401, 2.309401,
                                             2.041241, 2.041241, 2.041241, 2.041241, 2.041241, 2.041241});
  sd::ops::unsorted_segment_sqrt_n_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Hello Out:");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4}, {2.7577164, 3.4648232, 4.9497476, 12.727922, 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  sd::ops::unsorted_segment_sqrt_n op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    result.at(0)->printIndexedBuffer("Output");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4},
      {91.,       82.,      37.,       64.,      55.1,      46.4,     73.,       28.,       119.1,     12.1,
       112.7,     13.1,     14.,       114.2,    16.2,      117.,     57.982758, 45.254833, 108.89445, 48.083263,
       49.638893, 72.69058, 117.37973, 39.59798, 161.36177, 66.60946, 88.67119,  89.23688,  90.50967,  90.79251,
       93.62093,  90.50967, 91.,       82.,      37.,       64.,      55.1,      46.4,      73.,       28.,
       119.1,     12.1,     112.7,     13.1,     14.,       114.2,    16.2,      117.});

  sd::ops::unsorted_segment_sqrt_n op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::unsorted_segment_sqrt_n op;

  auto result = op.evaluate({&x, &idx}, {}, {8});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_5) {
  auto x = NDArrayFactory::create<double>({1., 2., 5., 7., 3., 1., 3., 4.});
  auto idx = NDArrayFactory::create<int>({3, 1, 0, 0, 2, 0, 3, 2});
  // NDArray<double> exp({1.7320508075688772, 1.,      1.4142135623730951,        1.4142135623730951});
  auto exp = NDArrayFactory::create<double>({7.5055537, 2., 4.9497476, 2.828427});
  sd::ops::unsorted_segment_sqrt_n op;

  auto result = op.evaluate({&x, &idx}, {}, {4});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSqrtN_6) {
  auto x = NDArrayFactory::create<double>({5, 1, 7, 2, 3, 4, 1, 3});
  auto idx = NDArrayFactory::create<int>({0, 0, 0, 1, 2, 2, 3, 3});
  // NDArray<double> exp({1.7320508075688772, 1.,      1.4142135623730951,        1.4142135623730951});
  //    auto exp = NDArrayFactory::create<double>({7.5055537, 2.,        4.9497476, 2.828427});
  sd::ops::unsorted_segment_sqrt_n op;

  try {
    auto result = op.evaluate({&x, &idx}, {}, {1});
    ASSERT_NE(result.status(), sd::Status::OK);
  } catch (std::exception& err) {
  }
  // result.at(0)->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");
  // ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentSum_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({4.3, 17.5, 3., 13.2, 11.2});

  sd::ops::segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentSumBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>({1., 1., 2., 2., 2., 2., 3., 4., 4., 4., 5., 5., 5., 5., 5., 5.});
  sd::ops::segment_sum_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSumBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1, 2, 3, 4, 5});
  auto exp = NDArrayFactory::create<double>({1., 1., 2., 2., 2., 2., 3., 4., 4., 4., 5., 5., 5., 5., 5., 5.});
  sd::ops::unsorted_segment_sum_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSumBP_2) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>({3., 1., 1., 2., 2., 2., 2., 4., 4., 4., 5., 5., 5., 5., 5., 5.});
  sd::ops::unsorted_segment_sum_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentSum_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {3.9, 4.9, 7., 18., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  sd::ops::segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentSumBP_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp =
      NDArrayFactory::create<double>('c', {4, 4}, {1., 2., 3., 4., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  auto eps = NDArrayFactory::create<double>('c', {3, 4});
  eps.linspace(1);

  sd::ops::segment_sum_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentSum_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4},
      {91., 82., 37.,  64., 55.1, 46.4,  73.,  28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
       82., 64., 154., 68., 70.2, 102.8, 166., 56., 228.2, 94.2, 125.4, 126.2, 128., 128.4, 132.4, 128.,
       91., 82., 37.,  64., 55.1, 46.4,  73.,  28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentSum_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSum_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({4.3, 17.5, 3., 13.2, 11.2});

  sd::ops::unsorted_segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSum_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {3.9, 4.9, 7., 18., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  sd::ops::unsorted_segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSum_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4},
      {91., 82., 37.,  64., 55.1, 46.4,  73.,  28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
       82., 64., 154., 68., 70.2, 102.8, 166., 56., 228.2, 94.2, 125.4, 126.2, 128., 128.4, 132.4, 128.,
       91., 82., 37.,  64., 55.1, 46.4,  73.,  28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::unsorted_segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentSum_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 3, 7});
  auto exp = NDArrayFactory::create<double>(
      'c', {8, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,    0.,   0.,    0.,    0.,   0.,    0.,    0.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  sd::ops::unsorted_segment_sum op;

  auto result = op.evaluate({&x, &idx}, {}, {8});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({4.5, 181.44, 3., 39.69, 1.9404});

  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProdBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>(
      {2.5, 1.8, 90.72, 40.32, 172.8, 151.2, 3., 17.64, 75.6, 75.6, 13.86, 97.02, 3.234, 2.31, 4.41, 9.702});
  sd::ops::segment_prod_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("ProdBP Output");
  //    exp.printIndexedBuffer("ProdBP Expect");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProdBP_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>(
      {2.5, 1.8, 90.72, 40.32, 172.8, 151.2, 3., 17.64, 75.6, 75.6, 13.86, 97.02, 3.234, 2.31, 4.41, 9.702});
  sd::ops::segment_prod_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("ProdBP Output");
  // exp.printIndexedBuffer("ProdBP Expect");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProdBP_2) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto eps = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
  auto exp = NDArrayFactory::create<double>(
      {3., 2.5, 1.8, 90.72, 40.32, 172.8, 151.2, 17.64, 75.6, 75.6, 13.86, 97.02, 3.234, 2.31, 4.41, 9.702});
  auto n = NDArrayFactory::create<sd::LongType>(5LL);
  sd::ops::unsorted_segment_prod_bp op;

  auto result = op.evaluate({&x, &idx, &eps, &n}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Unsorted ProdBP Output");
  // exp.printIndexedBuffer("Unsorted ProdBP Expect");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {3.78, 6., 12., 81., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProdBP_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto eps = NDArrayFactory::create<double>('c', {3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4, 4},
                                            {2.1, 4.8, 9., 36., 1.8, 5., 12., 36., 5., 6., 7., 8., 9., 10., 11., 12.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}
  eps.linspace(1);
  sd::ops::segment_prod_bp op;

  auto result = op.evaluate({&x, &idx, &eps}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4},
      {91.,       82.,       37.,       64.,     55.1,  46.4,    73.,       28.,  119.1,     12.1,      112.7, 13.1,
       14.,       114.2,     16.2,      117.,    1581,  924,     5829,      1056, 832.01001, 2616.9602, 6789,  784,
       12993.810, 993.41003, 1431.2899, 1481.61, 1596,  1621.64, 1882.4401, 1287, 91.,       82.,       37.,   64.,
       55.1,      46.4,      73.,       28.,     119.1, 12.1,    112.7,     13.1, 14.,       114.2,     16.2,  117.});

  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_04) {
  auto x = NDArrayFactory::create<int>({1, 2, 3, 4, 5, 6, 7, 8});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2, 2, 2, 3, 3});
  auto exp = NDArrayFactory::create<int>({2, 3, 120, 56});

  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_05) {
  auto x = NDArrayFactory::create<int16_t>({1, 2, 3, 4, 5, 6, 7, 8});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2, 2, 2, 3, 3});
  auto exp = NDArrayFactory::create<int16_t>({2, 3, 120, 56});

  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto res = result.at(0);
  //    res->printIndexedBuffer("Segment prod 05");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_05_1) {
  auto x = NDArrayFactory::create<int>({1, 2, 3, 4, 5, 6, 7, 8});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2, 2, 2, 3, 3});
  auto exp = NDArrayFactory::create<int>({2, 3, 120, 56});

  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto res = result.at(0);
  //    res->printIndexedBuffer("Segment prod 05_1");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_06) {
  auto x = NDArrayFactory::create<int8_t>({'\x1', '\x2', '\x3', '\x4', '\x5', '\x6', '\x7', '\x8'});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2, 2, 2, 3, 3});
  auto exp = NDArrayFactory::create<int8_t>({2, 3, 120, 56});
  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_07) {
  auto x = NDArrayFactory::create<uint8_t>({'\x1', '\x2', '\x3', '\x4', '\x5', '\x6', '\x7', '\x8'});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2, 2, 2, 3, 3});
  auto exp = NDArrayFactory::create<uint8_t>({2, 3, 120, 56});
  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestSegmentProd_08) {
  auto x = NDArrayFactory::create<int>({'\x1', '\x2', '\x3', '\x4', '\x5', '\x6', '\x7', '\x8', '\x9', '\xA'});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 2, 2, 2, 2, 3, 3, 3, 3});
  auto exp = NDArrayFactory::create<int>({2, 1, 360, 5040});
  sd::ops::segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_1) {
  auto x = NDArrayFactory::create<double>({1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({4.5, 181.44, 3., 39.69, 1.9404});

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_11) {
  auto x = NDArrayFactory::create<double>({3., 1.8, 2.5, 4., 9., 2.1, 2.4, 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 4});
  auto exp = NDArrayFactory::create<double>({4.5, 181.44, 3., 39.69, 1.9404});

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {5});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_2) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});
  auto idx = NDArrayFactory::create<int>({0, 0, 1, 2});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {3.78, 6., 12., 81., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_12) {
  auto x = NDArrayFactory::create<double>('c', {4, 4},
                                          {3., 4.2, 2.2, 1., 1.8, 2.5, 4., 9., 2.1, 2.4, 3., 9., 2.1, 2.1, 0.7, 0.1});
  auto idx = NDArrayFactory::create<int>({2, 0, 0, 1});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {3.78, 6., 12., 81., 2.1, 2.1, 0.7, 0.1, 3., 4.2, 2.2, 1.});

  //{ 2.1, 2.5,  4.,  9., 2.1, 2.1, 0.7, 0.1, 3.,  4.2, 2.2, 1.}

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 1);
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_08) {
  auto x = NDArrayFactory::create<int>({'\x1', '\x2', '\x3', '\x4', '\x5', '\x6', '\x7', '\x8', '\x9', '\xA'});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 2, 2, 2, 2, 3, 3, 3, 3});
  auto exp = NDArrayFactory::create<int>({2, 1, 360, 5040});
  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {4});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4},
      {91.,       82.,       37.,   64.,   55.1,      46.4,      73.,       28.,     119.1,     12.1,
       112.7,     13.1,      14.,   114.2, 16.2,      117.,      1581,      924,     5829,      1056,
       832.01001, 2616.9602, 6789,  784,   12993.810, 993.41003, 1431.2899, 1481.61, 1596.0000, 1621.6399,
       1882.4401, 1287,      91.,   82.,   37.,       64.,       55.1,      46.4,    73.,       28.,
       119.1,     12.1,      112.7, 13.1,  14.,       114.2,     16.2,      117.});

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {4, 4, 4}, {91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       51., 42., 67., 24., 15.1, 56.4, 93., 28., 109.1, 82.1, 12.7,  113.1, 114., 14.2,  116.2, 11.,
                       31., 22., 87., 44., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.,
                       91., 82., 37., 64., 55.1, 46.4, 73., 28., 119.1, 12.1, 112.7, 13.1,  14.,  114.2, 16.2,  117.});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({1, 1, 1, 2});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4, 4}, {1.,        1.,        1.,        1.,        1.,       1.,        1.,        1.,
                       1.,        1.,        1.,        1.,        1.,       1.,        1.,        1.,

                       143871,    75768,     215673,    67584.,    45843.75, 121426.96, 495597,    21952,
                       1547562.8, 12020.262, 161306.38, 19409.092, 22344,    185191.27, 30495.531, 150579,

                       91.,       82.,       37.,       64,        55.1,     46.400002, 73,        28,
                       119.1,     12.1,      112.7,     13.1,      14,       114.2,     16.2,      117});

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {3});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProd_5) {
  auto x = NDArrayFactory::create<double>('c', {8, 15});

  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({3, 1, 2, 1, 2, 3, 2, 1});
  auto exp = NDArrayFactory::create<double>(
      'c', {4, 15},
      {1.,      1.,      1.,      1.,      1.,      1.,      1.,      1.,      1.,      1.,      1.,      1.,
       1.,      1.,      1.,      78016.,  85493.,  93312.,  101479., 110000., 118881., 128128., 137747., 147744.,
       158125., 168896., 180063., 191632., 203609., 216000., 172081., 182528., 193347., 204544., 216125., 228096.,
       240463., 253232., 266409., 280000., 294011., 308448., 323317., 338624., 354375., 76.,     154.,    234.,
       316.,    400.,    486.,    574.,    664.,    756.,    850.,    946.,    1044.,   1144.,   1246.,   1350.});
  x.linspace(1.);

  sd::ops::unsorted_segment_prod op;

  auto result = op.evaluate({&x, &idx}, {}, {4});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  //    result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestUnsortedSegmentProdBP_4) {
  auto x = NDArrayFactory::create<double>('c', {8}, {5, 1, 7, 2, 3, 4, 1, 3});
  auto gradO = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 4});
  // ----------------------------------------------------------------

  auto idx = NDArrayFactory::create<int>({0, 0, 0, 1, 2, 2, 3, 3});
  auto exp = NDArrayFactory::create<double>(
      'c', {8}, {7.000000, 35.000000, 5.000000, 2.000000, 12.000000, 9.000000, 12.000000, 4.000000});
  //            1., 1., 1., 1.,      1., 1.,1.,1.,     1.,1.,1.,1.,     1.,1.,1.,1.,
  //
  //            143871, 75768, 215673, 67584.,     45843.75, 121426.96, 495597, 21952,
  //            1547562.8, 12020.262, 161306.38, 19409.092,  22344, 185191.27, 30495.531, 150579,
  //
  //            91., 82., 37., 64,     55.1, 46.400002, 73, 28,    119.1, 12.1, 112.7, 13.1,    14, 114.2, 16.2, 117});

  sd::ops::unsorted_segment_prod_bp op;

  auto result = op.evaluate({&x, &idx, &gradO}, {}, {4});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  // exp.printIndexedBuffer("Expect");
  //    exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_1) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 4, 4, 4},
      {91., 82.,  37.,  64.,  55.,  46.,  73.,  28.,  119., 12.,  112., 13.,  14.,  114., 16.,  117., 51.,  42.,  67.,
       24., 15.,  56.,  93.,  28.,  109., 82.,  12.,  113., 114., 14.,  116., 11.,  31.,  22.,  87.,  44.,  55.,  46.,
       73., 28.,  119., 12.,  112., 13.,  14.,  114., 16.,  117., 91.,  82.,  37.,  64.,  55.1, 46.4, 73.,  28.,  119.,
       12., 112., 13.,  14.,  114., 16.2, 117., 91.,  82.,  37.,  64.,  55.,  46.,  73.,  28.,  119., 12.,  112., 13.,
       14., 114., 16.,  117., 51.,  42.,  67.,  24.,  15.,  56.,  93.,  28.,  109., 82.,  12.,  113., 114., 14.,  116.,
       11., 31.,  22.,  87.,  44.,  55.,  46.,  73.,  28.,  119., 12.,  112., 13.,  14.,  114., 16.,  117., 91.,  82.,
       37., 64.,  55.1, 46.4, 73.,  28.,  119., 12.,  112., 13.,  140., 110., 160., 107.});

  // ----------------------------------------------------------------

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 4, 4, 4},
      {91., 82.,  37.,  64.,  55.,  46.,  73.,  28.,  119., 12.,  112., 13.,  14.,  114., 16.,  117., 51.,  42.,  67.,
       24., 15.,  56.,  93.,  28.,  109., 82.,  12.,  113., 114., 14.,  116., 11.,  31.,  22.,  87.,  44.,  55.,  46.,
       73., 28.,  119., 12.,  112., 13.,  14.,  114., 16.,  117., 91.,  82.,  37.,  64.,  55.1, 46.4, 73.,  28.,  119.,
       12., 112., 13.,  14.,  114., 16.2, 117., 91.,  82.,  37.,  64.,  55.,  46.,  73.,  28.,  119., 12.,  112., 13.,
       14., 114., 16.,  117., 51.,  42.,  67.,  24.,  15.,  56.,  93.,  28.,  109., 82.,  12.,  113., 114., 14.,  116.,
       11., 31.,  22.,  87.,  44.,  55.,  46.,  73.,  28.,  119., 12.,  112., 13.,  14.,  114., 16.,  117., 91.,  82.,
       37., 64.,  55.1, 46.4, 73.,  28.,  119., 12.,  112., 13.,  140., 110., 160., 107.});

  sd::ops::extract_image_patches op;

  auto result = op.evaluate({&x}, {}, {1, 1, 1, 1, 1, 1, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_2) {
  auto x = NDArrayFactory::create<double>(
      'c', {3, 3, 4, 3},
      {11.,  12., 13.,  12., 13.,  14., 15., 16.,  17., 18.,  19., 10., 1.,   2.,   3.,   2.,   3.,   4.,
       21.,  22., 23.,  22., 23.,  24., 5.,  6.,   7.,  8.,   9.,  0.,  35.,  36.,  37.,  38.,  39.,  40.,
       9.,   8.,  7.,   6.,  5.,   4.,  49., 48.,  47., 46.,  45., 44., 3.,   2.,   1.,   0.,   1.,   2.,
       53.,  52., 51.,  50., 51.,  52., 15., 16.,  17., 18.,  19., 10., 135., 136., 137., 138., 139., 140.,
       211., 12., 13.,  12., 213., 14., 15., 216., 17., 128., 19., 10., 21.,  2.,   3.,   2.,   3.,   24.,
       21.,  22., 223., 22., 223., 24., 25., 6.,   7.,  8.,   9.,  20., 35.,  36.,  327., 38.,  239., 40.});

  // Images shape is  (3, 3, 4, 3)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 1, 1, 12}, {11., 12., 13., 12., 13., 14., 1.,   2.,  3.,  2.,  3.,   4.,  9.,  8., 7., 6., 5., 4.,
                           3.,  2.,  1.,  0.,  1.,  2.,  211., 12., 13., 12., 213., 14., 21., 2., 3., 2., 3., 24.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate({&x}, {}, {2, 2, 3, 3, 1, 1, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);

  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {3, 3, 4, 3},
      {11.,  12., 13.,  12., 13.,  14., 15., 16.,  17., 18.,  19., 10., 1.,   2.,   3.,   2.,   3.,   4.,
       21.,  22., 23.,  22., 23.,  24., 5.,  6.,   7.,  8.,   9.,  0.,  35.,  36.,  37.,  38.,  39.,  40.,
       9.,   8.,  7.,   6.,  5.,   4.,  49., 48.,  47., 46.,  45., 44., 3.,   2.,   1.,   0.,   1.,   2.,
       53.,  52., 51.,  50., 51.,  52., 15., 16.,  17., 18.,  19., 10., 135., 136., 137., 138., 139., 140.,
       211., 12., 13.,  12., 213., 14., 15., 216., 17., 128., 19., 10., 21.,  2.,   3.,   2.,   3.,   24.,
       21.,  22., 223., 22., 223., 24., 25., 6.,   7.,  8.,   9.,  20., 35.,  36.,  327., 38.,  239., 40.});

  // Images shape is  (3, 3, 4, 3)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 1, 2, 6},
      {11., 12., 13., 5.,   6.,   7.,   15.,  16., 17., 35., 36., 37., 9.,  8.,   7.,  15., 16., 17.,
       49., 48., 47., 135., 136., 137., 211., 12., 13., 25., 6.,  7.,  15., 216., 17., 35., 36., 327.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate({&x}, {}, {2, 1, 3, 2, 2, 2, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);

  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {3, 3, 4, 3},
      {11.,  12., 13.,  12., 13.,  14., 15., 16.,  17., 18.,  19., 10., 1.,   2.,   3.,   2.,   3.,   4.,
       21.,  22., 23.,  22., 23.,  24., 5.,  6.,   7.,  8.,   9.,  0.,  35.,  36.,  37.,  38.,  39.,  40.,
       9.,   8.,  7.,   6.,  5.,   4.,  49., 48.,  47., 46.,  45., 44., 3.,   2.,   1.,   0.,   1.,   2.,
       53.,  52., 51.,  50., 51.,  52., 15., 16.,  17., 18.,  19., 10., 135., 136., 137., 138., 139., 140.,
       211., 12., 13.,  12., 213., 14., 15., 216., 17., 128., 19., 10., 21.,  2.,   3.,   2.,   3.,   24.,
       21.,  22., 223., 22., 223., 24., 25., 6.,   7.,  8.,   9.,  20., 35.,  36.,  327., 38.,  239., 40.});

  // Images shape is  (3, 3, 4, 3)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 3, 4, 3},
      {11.,  12., 13.,  12., 13.,  14., 15., 16.,  17., 18.,  19., 10., 1.,   2.,   3.,   2.,   3.,   4.,
       21.,  22., 23.,  22., 23.,  24., 5.,  6.,   7.,  8.,   9.,  0.,  35.,  36.,  37.,  38.,  39.,  40.,
       9.,   8.,  7.,   6.,  5.,   4.,  49., 48.,  47., 46.,  45., 44., 3.,   2.,   1.,   0.,   1.,   2.,
       53.,  52., 51.,  50., 51.,  52., 15., 16.,  17., 18.,  19., 10., 135., 136., 137., 138., 139., 140.,
       211., 12., 13.,  12., 213., 14., 15., 216., 17., 128., 19., 10., 21.,  2.,   3.,   2.,   3.,   24.,
       21.,  22., 223., 22., 223., 24., 25., 6.,   7.,  8.,   9.,  20., 35.,  36.,  327., 38.,  239., 40.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate({&x}, {}, {1, 1, 1, 1, 1, 1, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_5) {
  auto x = NDArrayFactory::create<double>(
      'c', {3, 3, 4, 3},
      {11.,  12., 13.,  12., 13.,  14., 15., 16.,  17., 18.,  19., 10., 1.,   2.,   3.,   2.,   3.,   4.,
       21.,  22., 23.,  22., 23.,  24., 5.,  6.,   7.,  8.,   9.,  0.,  35.,  36.,  37.,  38.,  39.,  40.,
       9.,   8.,  7.,   6.,  5.,   4.,  49., 48.,  47., 46.,  45., 44., 3.,   2.,   1.,   0.,   1.,   2.,
       53.,  52., 51.,  50., 51.,  52., 15., 16.,  17., 18.,  19., 10., 135., 136., 137., 138., 139., 140.,
       211., 12., 13.,  12., 213., 14., 15., 216., 17., 128., 19., 10., 21.,  2.,   3.,   2.,   3.,   24.,
       21.,  22., 223., 22., 223., 24., 25., 6.,   7.,  8.,   9.,  20., 35.,  36.,  327., 38.,  239., 40.});

  // Images shape is  (3, 3, 4, 3)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 1, 1, 18},
      {
          11., 12., 13.,  15., 16., 17., 1., 2.,  3.,  21.,  22., 23., 5.,  6.,  7.,   35.,  36.,  37.,  9.,
          8.,  7.,  49.,  48., 47., 3.,  2., 1.,  53., 52.,  51., 15., 16., 17., 135., 136., 137., 211., 12.,
          13., 15., 216., 17., 21., 2.,  3., 21., 22., 223., 25., 6.,  7.,  35., 36.,  327.

          // Patch shape is (3, 1, 2, 18)

      });
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate({&x}, {}, {3, 2, 3, 2, 1, 2, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);
  //    result.at(0)->printIndexedBuffer("Output");
  // result.at(0)->printShapeInfo("Out Shape");
  //    exp.printIndexedBuffer("Expect");
  // exp.printShapeInfo("Exp Shape");
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_6) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
       21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42});

  // Images shape is  (3, 3, 4, 3)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {2, 1, 4, 4},
      {11.11, 11.12, 12.11, 12.12, 11.21, 11.22, 12.21, 12.22, 11.31, 11.32, 12.31, 12.32, 11.41, 11.42, 12.41, 12.42,
       21.11, 21.12, 22.11, 22.12, 21.21, 21.22, 22.21, 22.22, 21.31, 21.32, 22.31, 22.32, 21.41, 21.42, 22.41, 22.42});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate({&x}, {}, {2, 1, 1, 1, 1, 1, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.isSameShape(result.at(0)));
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_7) {
  auto x = NDArrayFactory::create<double>('c', {1, 3, 3, 1});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 3, 4},
                                            {1., 2., 4., 5., 2., 3., 5., 6., 3., 0., 6., 0., 4., 5., 7., 8., 5., 6.,
                                             8., 9., 6., 0., 9., 0., 7., 8., 0., 0., 8., 9., 0., 0., 9., 0., 0., 0.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("Output");
  //    exp.printBuffer("Expect");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //        printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_8) {
  auto x = NDArrayFactory::create<double>('c', {1, 3, 3, 2});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 3, 3, 8}, {1,  2,  3,  4,  7,  8,  9,  10, 3,  4,  5,  6,  9,  10, 11, 12, 5,  6,  0, 0, 11, 12, 0, 0,
                          7,  8,  9,  10, 13, 14, 15, 16, 9,  10, 11, 12, 15, 16, 17, 18, 11, 12, 0, 0, 17, 18, 0, 0,
                          13, 14, 15, 16, 0,  0,  0,  0,  15, 16, 17, 18, 0,  0,  0,  0,  17, 18, 0, 0, 0,  0,  0, 0});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("Output");
  //    exp.printBuffer("Expect");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_9) {
  auto x = NDArrayFactory::create<double>('c', {1, 6, 6, 2});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 6, 6, 18},
      {0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  4.,  0.,  0.,  13., 14., 15., 16., 0.,  0.,  0.,  0.,
       0.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  13., 14., 15., 16., 17., 18., 0.,  0.,  0.,  0.,  0.,  0.,  3.,  4.,
       5.,  6.,  7.,  8.,  15., 16., 17., 18., 19., 20., 0.,  0.,  0.,  0.,  0.,  0.,  5.,  6.,  7.,  8.,  9.,  10.,
       17., 18., 19., 20., 21., 22., 0.,  0.,  0.,  0.,  0.,  0.,  7.,  8.,  9.,  10., 11., 12., 19., 20., 21., 22.,
       23., 24., 0.,  0.,  0.,  0.,  0.,  0.,  9.,  10., 11., 12., 0.,  0.,  21., 22., 23., 24., 0.,  0.,  0.,  0.,
       1.,  2.,  3.,  4.,  0.,  0.,  13., 14., 15., 16., 0.,  0.,  25., 26., 27., 28., 1.,  2.,  3.,  4.,  5.,  6.,
       13., 14., 15., 16., 17., 18., 25., 26., 27., 28., 29., 30., 3.,  4.,  5.,  6.,  7.,  8.,  15., 16., 17., 18.,
       19., 20., 27., 28., 29., 30., 31., 32., 5.,  6.,  7.,  8.,  9.,  10., 17., 18., 19., 20., 21., 22., 29., 30.,
       31., 32., 33., 34., 7.,  8.,  9.,  10., 11., 12., 19., 20., 21., 22., 23., 24., 31., 32., 33., 34., 35., 36.,
       9.,  10., 11., 12., 0.,  0.,  21., 22., 23., 24., 0.,  0.,  33., 34., 35., 36., 0.,  0.,  0.,  0.,  13., 14.,
       15., 16., 0.,  0.,  25., 26., 27., 28., 0.,  0.,  37., 38., 39., 40., 13., 14., 15., 16., 17., 18., 25., 26.,
       27., 28., 29., 30., 37., 38., 39., 40., 41., 42., 15., 16., 17., 18., 19., 20., 27., 28., 29., 30., 31., 32.,
       39., 40., 41., 42., 43., 44., 17., 18., 19., 20., 21., 22., 29., 30., 31., 32., 33., 34., 41., 42., 43., 44.,
       45., 46., 19., 20., 21., 22., 23., 24., 31., 32., 33., 34., 35., 36., 43., 44., 45., 46., 47., 48., 21., 22.,
       23., 24., 0.,  0.,  33., 34., 35., 36., 0.,  0.,  45., 46., 47., 48., 0.,  0.,  0.,  0.,  25., 26., 27., 28.,
       0.,  0.,  37., 38., 39., 40., 0.,  0.,  49., 50., 51., 52., 25., 26., 27., 28., 29., 30., 37., 38., 39., 40.,
       41., 42., 49., 50., 51., 52., 53., 54., 27., 28., 29., 30., 31., 32., 39., 40., 41., 42., 43., 44., 51., 52.,
       53., 54., 55., 56., 29., 30., 31., 32., 33., 34., 41., 42., 43., 44., 45., 46., 53., 54., 55., 56., 57., 58.,
       31., 32., 33., 34., 35., 36., 43., 44., 45., 46., 47., 48., 55., 56., 57., 58., 59., 60., 33., 34., 35., 36.,
       0.,  0.,  45., 46., 47., 48., 0.,  0.,  57., 58., 59., 60., 0.,  0.,  0.,  0.,  37., 38., 39., 40., 0.,  0.,
       49., 50., 51., 52., 0.,  0.,  61., 62., 63., 64., 37., 38., 39., 40., 41., 42., 49., 50., 51., 52., 53., 54.,
       61., 62., 63., 64., 65., 66., 39., 40., 41., 42., 43., 44., 51., 52., 53., 54., 55., 56., 63., 64., 65., 66.,
       67., 68., 41., 42., 43., 44., 45., 46., 53., 54., 55., 56., 57., 58., 65., 66., 67., 68., 69., 70., 43., 44.,
       45., 46., 47., 48., 55., 56., 57., 58., 59., 60., 67., 68., 69., 70., 71., 72., 45., 46., 47., 48., 0.,  0.,
       57., 58., 59., 60., 0.,  0.,  69., 70., 71., 72., 0.,  0.,  0.,  0.,  49., 50., 51., 52., 0.,  0.,  61., 62.,
       63., 64., 0.,  0.,  0.,  0.,  0.,  0.,  49., 50., 51., 52., 53., 54., 61., 62., 63., 64., 65., 66., 0.,  0.,
       0.,  0.,  0.,  0.,  51., 52., 53., 54., 55., 56., 63., 64., 65., 66., 67., 68., 0.,  0.,  0.,  0.,  0.,  0.,
       53., 54., 55., 56., 57., 58., 65., 66., 67., 68., 69., 70., 0.,  0.,  0.,  0.,  0.,  0.,  55., 56., 57., 58.,
       59., 60., 67., 68., 69., 70., 71., 72., 0.,  0.,  0.,  0.,  0.,  0.,  57., 58., 59., 60., 0.,  0.,  69., 70.,
       71., 72., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {3, 3, 1, 1, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("OutputSame");
  //    exp.printBuffer("ExpectSame");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_9_1) {
  auto x = NDArrayFactory::create<double>('c', {1, 4, 4, 2},
                                          {1, 116, 2,  116, 3,  116, 4,  116, 5,  117, 6,  117, 7,  117, 8,  117,
                                           9, 118, 10, 118, 11, 118, 12, 118, 13, 119, 14, 119, 15, 119, 16, 119});
  // x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 4, 4, 8},
      {1,  116, 2,  116, 5,  117, 6,  117, 2,  116, 3,  116, 6,  117, 7,  117, 3,  116, 4,  116, 7,  117,
       8,  117, 4,  116, 0,  0,   8,  117, 0,  0,   5,  117, 6,  117, 9,  118, 10, 118, 6,  117, 7,  117,
       10, 118, 11, 118, 7,  117, 8,  117, 11, 118, 12, 118, 8,  117, 0,  0,   12, 118, 0,  0,   9,  118,
       10, 118, 13, 119, 14, 119, 10, 118, 11, 118, 14, 119, 15, 119, 11, 118, 12, 118, 15, 119, 16, 119,
       12, 118, 0,  0,   16, 119, 0,  0,   13, 119, 14, 119, 0,  0,   0,  0,   14, 119, 15, 119, 0,  0,
       0,  0,   15, 119, 16, 119, 0,  0,   0,  0,   16, 119, 0,  0,   0,  0,   0,  0

      });
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("OutputSame");
  //    exp.printBuffer("ExpectSame");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

//
//
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_10) {
  auto x = NDArrayFactory::create<double>('c', {1, 6, 6, 2});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 4, 4, 18},
      {1.,  2.,  3.,  4.,  5.,  6.,  13., 14., 15., 16., 17., 18., 25., 26., 27., 28., 29., 30., 3.,  4.,  5.,
       6.,  7.,  8.,  15., 16., 17., 18., 19., 20., 27., 28., 29., 30., 31., 32., 5.,  6.,  7.,  8.,  9.,  10.,
       17., 18., 19., 20., 21., 22., 29., 30., 31., 32., 33., 34., 7.,  8.,  9.,  10., 11., 12., 19., 20., 21.,
       22., 23., 24., 31., 32., 33., 34., 35., 36., 13., 14., 15., 16., 17., 18., 25., 26., 27., 28., 29., 30.,
       37., 38., 39., 40., 41., 42., 15., 16., 17., 18., 19., 20., 27., 28., 29., 30., 31., 32., 39., 40., 41.,
       42., 43., 44., 17., 18., 19., 20., 21., 22., 29., 30., 31., 32., 33., 34., 41., 42., 43., 44., 45., 46.,
       19., 20., 21., 22., 23., 24., 31., 32., 33., 34., 35., 36., 43., 44., 45., 46., 47., 48., 25., 26., 27.,
       28., 29., 30., 37., 38., 39., 40., 41., 42., 49., 50., 51., 52., 53., 54., 27., 28., 29., 30., 31., 32.,
       39., 40., 41., 42., 43., 44., 51., 52., 53., 54., 55., 56., 29., 30., 31., 32., 33., 34., 41., 42., 43.,
       44., 45., 46., 53., 54., 55., 56., 57., 58., 31., 32., 33., 34., 35., 36., 43., 44., 45., 46., 47., 48.,
       55., 56., 57., 58., 59., 60., 37., 38., 39., 40., 41., 42., 49., 50., 51., 52., 53., 54., 61., 62., 63.,
       64., 65., 66., 39., 40., 41., 42., 43., 44., 51., 52., 53., 54., 55., 56., 63., 64., 65., 66., 67., 68.,
       41., 42., 43., 44., 45., 46., 53., 54., 55., 56., 57., 58., 65., 66., 67., 68., 69., 70., 43., 44., 45.,
       46., 47., 48., 55., 56., 57., 58., 59., 60., 67., 68., 69., 70., 71., 72.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;
  // x.printIndexedBuffer("Images");
  // x.printBuffer("Images linear");
  auto result = op.evaluate(
      {&x}, {},
      {3, 3, 1, 1, 1, 1, 0});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("OutputValid");
  //    exp.printBuffer("ExpectValid");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_010) {
  auto x = NDArrayFactory::create<double>('c', {1, 4, 4, 1});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 3, 4},
                                            {1,  2,  5, 6, 2,  3,  6, 7,  3,  4,  7,  8,  5,  6,  9,  10, 6,  7,
                                             10, 11, 7, 8, 11, 12, 9, 10, 13, 14, 10, 11, 14, 15, 11, 12, 15, 16});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;
  // x.printIndexedBuffer("Images");
  // x.printBuffer("Images linear");
  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 1, 1, 0});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("OutputValid");
  //    exp.printBuffer("ExpectValid");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_010_1) {
  auto x = NDArrayFactory::create<double>('c', {1, 4, 4, 1});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 4, 4, 4}, {1,  2,  5,  6, 2,  3,  6, 7, 3,  4,  7, 8,  4,  0,  8,  0,  5,  6,  9,  10, 6,  7,
                          10, 11, 7,  8, 11, 12, 8, 0, 12, 0,  9, 10, 13, 14, 10, 11, 14, 15, 11, 12, 15, 16,
                          12, 0,  16, 0, 13, 14, 0, 0, 14, 15, 0, 0,  15, 16, 0,  0,  16, 0,  0,  0});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;
  // x.printIndexedBuffer("Images");
  // x.printBuffer("Images linear");
  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("OutputSame");
  //    exp.printBuffer("ExpectSame");
  //    exp.printIndexedBuffer("Expect Same Formatted");
  //    output->printIndexedBuffer("Output Same Formatted");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_011) {
  auto x = NDArrayFactory::create<double>('c', {1, 4, 4, 1});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 4},
                                            {
                                                1,
                                                3,
                                                9,
                                                11,
                                                2,
                                                4,
                                                10,
                                                12,
                                                5,
                                                7,
                                                13,
                                                15,
                                                6,
                                                8,
                                                14,
                                                16,
                                            });
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;
  // x.printIndexedBuffer("Images");
  // x.printBuffer("Images linear");
  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 2, 2, 0});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("OutputValid");
  //    exp.printBuffer("ExpectValid");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_11) {
  auto x = NDArrayFactory::create<double>('c', {1, 8, 8, 2});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 4, 4, 8},
      {1,   2,   3,   4,   17,  18,  19,  20,  5,   6,   7,   8,   21,  22,  23,  24,  9,   10,  11,  12,  25,  26,
       27,  28,  13,  14,  15,  16,  29,  30,  31,  32,  33,  34,  35,  36,  49,  50,  51,  52,  37,  38,  39,  40,
       53,  54,  55,  56,  41,  42,  43,  44,  57,  58,  59,  60,  45,  46,  47,  48,  61,  62,  63,  64,  65,  66,
       67,  68,  81,  82,  83,  84,  69,  70,  71,  72,  85,  86,  87,  88,  73,  74,  75,  76,  89,  90,  91,  92,
       77,  78,  79,  80,  93,  94,  95,  96,  97,  98,  99,  100, 113, 114, 115, 116, 101, 102, 103, 104, 117, 118,
       119, 120, 105, 106, 107, 108, 121, 122, 123, 124, 109, 110, 111, 112, 125, 126, 127, 128});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 2, 2, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  //    output->printBuffer("Output");
  //    exp.printBuffer("Expect");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_12) {
  auto x = NDArrayFactory::create<double>('c', {1, 8, 8, 2});
  x.linspace(1);

  // Images shape is  (1, 3, 3, 4)
  //[1, 1, 1, 1]
  //[1, 3, 2, 1]
  auto exp = NDArrayFactory::create<double>(
      'c', {1, 8, 8, 8},
      {0,   0,   0,   0,   0,   0,   19,  20,  0,   0,   0,  0,   17,  18,  21,  22,  0,   0,   0,  0,   19,  20,  23,
       24,  0,   0,   0,   0,   21,  22,  25,  26,  0,   0,  0,   0,   23,  24,  27,  28,  0,   0,  0,   0,   25,  26,
       29,  30,  0,   0,   0,   0,   27,  28,  31,  32,  0,  0,   0,   0,   29,  30,  0,   0,   0,  0,   3,   4,   0,
       0,   35,  36,  1,   2,   5,   6,   33,  34,  37,  38, 3,   4,   7,   8,   35,  36,  39,  40, 5,   6,   9,   10,
       37,  38,  41,  42,  7,   8,   11,  12,  39,  40,  43, 44,  9,   10,  13,  14,  41,  42,  45, 46,  11,  12,  15,
       16,  43,  44,  47,  48,  13,  14,  0,   0,   45,  46, 0,   0,   0,   0,   19,  20,  0,   0,  51,  52,  17,  18,
       21,  22,  49,  50,  53,  54,  19,  20,  23,  24,  51, 52,  55,  56,  21,  22,  25,  26,  53, 54,  57,  58,  23,
       24,  27,  28,  55,  56,  59,  60,  25,  26,  29,  30, 57,  58,  61,  62,  27,  28,  31,  32, 59,  60,  63,  64,
       29,  30,  0,   0,   61,  62,  0,   0,   0,   0,   35, 36,  0,   0,   67,  68,  33,  34,  37, 38,  65,  66,  69,
       70,  35,  36,  39,  40,  67,  68,  71,  72,  37,  38, 41,  42,  69,  70,  73,  74,  39,  40, 43,  44,  71,  72,
       75,  76,  41,  42,  45,  46,  73,  74,  77,  78,  43, 44,  47,  48,  75,  76,  79,  80,  45, 46,  0,   0,   77,
       78,  0,   0,   0,   0,   51,  52,  0,   0,   83,  84, 49,  50,  53,  54,  81,  82,  85,  86, 51,  52,  55,  56,
       83,  84,  87,  88,  53,  54,  57,  58,  85,  86,  89, 90,  55,  56,  59,  60,  87,  88,  91, 92,  57,  58,  61,
       62,  89,  90,  93,  94,  59,  60,  63,  64,  91,  92, 95,  96,  61,  62,  0,   0,   93,  94, 0,   0,   0,   0,
       67,  68,  0,   0,   99,  100, 65,  66,  69,  70,  97, 98,  101, 102, 67,  68,  71,  72,  99, 100, 103, 104, 69,
       70,  73,  74,  101, 102, 105, 106, 71,  72,  75,  76, 103, 104, 107, 108, 73,  74,  77,  78, 105, 106, 109, 110,
       75,  76,  79,  80,  107, 108, 111, 112, 77,  78,  0,  0,   109, 110, 0,   0,   0,   0,   83, 84,  0,   0,   115,
       116, 81,  82,  85,  86,  113, 114, 117, 118, 83,  84, 87,  88,  115, 116, 119, 120, 85,  86, 89,  90,  117, 118,
       121, 122, 87,  88,  91,  92,  119, 120, 123, 124, 89, 90,  93,  94,  121, 122, 125, 126, 91, 92,  95,  96,  123,
       124, 127, 128, 93,  94,  0,   0,   125, 126, 0,   0,  0,   0,   99,  100, 0,   0,   0,   0,  97,  98,  101, 102,
       0,   0,   0,   0,   99,  100, 103, 104, 0,   0,   0,  0,   101, 102, 105, 106, 0,   0,   0,  0,   103, 104, 107,
       108, 0,   0,   0,   0,   105, 106, 109, 110, 0,   0,  0,   0,   107, 108, 111, 112, 0,   0,  0,   0,   109, 110,
       0,   0,   0,   0,   0,   0});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 2, 2, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,2,2,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  // output->printShapeInfo("Output shape");
  //    output->printIndexedBuffer("Output");
  //    exp.printBuffer("Expect");
  //    for (sd::LongType e = 0; e < exp.lengthOf(); e++)
  //        if (exp.e<double>(e) != output->e<double>(e))
  //            printf("%lld ", e);
  //    printf("\n");
  // result.at(1)->printBuffer("OUtput2");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestExtractImagePatches_SGO_13) {
  auto x = NDArrayFactory::create<double>('c', {1, 3, 3, 2});
  x.linspace(1);

  auto exp = NDArrayFactory::create<double>(
      'c', {1, 3, 3, 8}, {1.,  2.,  3.,  4.,  7.,  8.,  9., 10., 3.,  4.,  5.,  6.,  9.,  10., 11., 12., 5.,  6.,
                          0.,  0.,  11., 12., 0.,  0.,  7., 8.,  9.,  10., 13., 14., 15., 16., 9.,  10., 11., 12.,
                          15., 16., 17., 18., 11., 12., 0., 0.,  17., 18., 0.,  0.,  13., 14., 15., 16., 0.,  0.,
                          0.,  0.,  15., 16., 17., 18., 0., 0.,  0.,  0.,  17., 18., 0.,  0.,  0.,  0.,  0.,  0.});
  // ----------------------------------------------------------------
  sd::ops::extract_image_patches op;

  auto result = op.evaluate(
      {&x}, {},
      {2, 2, 1, 1, 1, 1, 1});  // equiv TF ksizes=[1,2,2,1], strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME"
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_1) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
       21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12,
       12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {6});
  ASSERT_EQ(result.status(), sd::Status::OK);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_2) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
       21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42,
       22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {-8});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_3) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
       21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42,
       22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {-40});
  ASSERT_EQ(result.status(), sd::Status::OK);

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_4) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
       21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12,
       12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {38});
  ASSERT_EQ(result.status(), sd::Status::OK);
  // result.at(0)->printIndexedBuffer("Output 4");
  // exp.printIndexedBuffer("Expect 4");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_4_inplace) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12, 12.21, 12.22, 12.31, 12.32, 12.41, 12.42,
       21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12, 22.21, 22.22, 22.31, 22.32, 22.41, 22.42});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 2, 4, 2},
      {22.21, 22.22, 22.31, 22.32, 22.41, 22.42, 11.11, 11.12, 11.21, 11.22, 11.31, 11.32, 11.41, 11.42, 12.11, 12.12,
       12.21, 12.22, 12.31, 12.32, 12.41, 12.42, 21.11, 21.12, 21.21, 21.22, 21.31, 21.32, 21.41, 21.42, 22.11, 22.12});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  NDArray* y = nullptr;
  auto result = op.execute({&x}, {y}, {}, {38}, {}, {}, true);
  ASSERT_EQ(result, sd::Status::OK);
  // x.printIndexedBuffer("Output 4 inplace");
  // exp.printIndexedBuffer("Expect 4 inplace");

  ASSERT_TRUE(exp.equalsTo(&x));

  //
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_5) {
  auto x = NDArrayFactory::create<double>('c', {3, 4}, {0., 1., 2., 3., 4, 5., 6., 7., 8., 9., 10., 11.});

  auto exp = NDArrayFactory::create<double>('c', {3, 4},
                                            {
                                                2., 3., 0., 1., 6., 7., 4., 5., 10., 11., 8., 9.
                                                //     4,  5,  6,  7, 8,  9, 10, 11, 0,  1,  2,  3
                                            });
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {2, 1});
  ASSERT_EQ(result.status(), sd::Status::OK);

  // result.at(0)->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2}, {0., 1., 2., 3., 4, 5., 6., 7., 8., 9., 10., 11.});

  auto exp = NDArrayFactory::create<double>('c', {2, 3, 2}, {1., 0., 3., 2., 5., 4., 7., 6., 9., 8., 11., 10.});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {1, 2});
  ASSERT_EQ(result.status(), sd::Status::OK);

  // result.at(0)->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2}, {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});

  auto exp = NDArrayFactory::create<double>('c', {2, 3, 2}, {11., 10., 7., 6., 9., 8., 5., 4., 1., 0., 3., 2.});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x}, {}, {1, 2, 1, 0});
  ASSERT_EQ(result.status(), sd::Status::OK);

  // result.at(0)->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(result.at(0)));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_8) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2}, {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});

  auto exp = NDArrayFactory::create<double>('c', {2, 3, 2}, {11., 10., 7., 6., 9., 8., 5., 4., 1., 0., 3., 2.});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  NDArray* y = nullptr;
  auto result = op.execute({&x}, {y}, {}, {1, 2, 1, 0}, {}, {}, true);
  ASSERT_EQ(result, sd::Status::OK);

  // x.printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(&x));

  //
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_9) {
  auto x = NDArrayFactory::create<double>(
      'c', {2, 3, 3}, {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 3}, {6., 7., 8., 0., 1., 2., 3., 4., 5., 15., 16., 17., 9., 10., 11., 12., 13., 14.});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  NDArray* y = nullptr;
  auto result = op.execute({&x}, {y}, {}, {1, 1}, {}, {}, true);
  ASSERT_EQ(result, sd::Status::OK);

  ASSERT_TRUE(exp.equalsTo(&x));

  //
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_10) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                           13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 4},
      {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  auto result = op.evaluate({&x}, {}, {3, 1});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  //    out->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_11) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                           13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  auto shift = NDArrayFactory::create<int>({1, 2});
  auto axis = NDArrayFactory::create<int>({0, 1});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {17., 18., 19., 20., 21., 22., 23., 24., 13., 14., 15., 16.,
                                                             5.,  6.,  7,   8,   9,   10,  11,  12,  1,   2,   3,   4});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  NDArray* y = nullptr;
  auto result = op.evaluate({&x, &shift, &axis});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  //    out->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_12) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                           13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  auto shift = NDArrayFactory::create<int>({1, 1, 1});
  auto axis = NDArrayFactory::create<int>({0, 1, 2});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 4}, {24, 21, 22, 23, 16, 13, 14, 15, 20, 17, 18, 19, 12, 9, 10, 11, 4, 1, 2, 3, 8, 5, 6, 7});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  NDArray* y = nullptr;
  auto result = op.evaluate({&x, &shift, &axis});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_13) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                           13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  auto shift = NDArrayFactory::create<int>(3);
  auto axis = NDArrayFactory::create<int>(2);

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 4}, {2, 3, 4, 1, 6, 7, 8, 5, 10, 11, 12, 9, 14, 15, 16, 13, 18, 19, 20, 17, 22, 23, 24, 21});
  // ----------------------------------------------------------------
  sd::ops::roll op;
  NDArray* y = nullptr;
  auto result = op.evaluate({&x}, {}, {3, 2});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);

  ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_14) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                                           13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  auto shift = NDArrayFactory::create<int>({1, 1, 1});
  auto axis = NDArrayFactory::create<int>({0, 1, 2});

  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 4}, {24, 21, 22, 23, 16, 13, 14, 15, 20, 17, 18, 19, 12, 9, 10, 11, 4, 1, 2, 3, 8, 5, 6, 7});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x, &shift, &axis});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);
  //    out->printIndexedBuffer("Output");
  // exp.printIndexedBuffer("Expect");

  ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TestRoll_15) {
  auto x = NDArrayFactory::create<float>({0.7788f, 0.8012f, 0.7244f, 0.2309f});
  auto shift = NDArrayFactory::create<int>(2);
  auto axis = NDArrayFactory::create<int>(0);

  auto exp = NDArrayFactory::create<float>({0.7244f, 0.2309f, 0.7788f, 0.8012f});
  // ----------------------------------------------------------------
  sd::ops::roll op;

  auto result = op.evaluate({&x, &shift, &axis});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto out = result.at(0);
  //    out->printIndexedBuffer("Output 15");
  //    exp.printIndexedBuffer("Expect 15");

  ASSERT_TRUE(exp.equalsTo(out));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test1) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});
  auto expected = NDArrayFactory::create<double>(50.);

  sd::ops::percentile op;

  auto result = op.evaluate({&input}, {50.}, {});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test2) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});
  auto expected = NDArrayFactory::create<double>('c', {1, 1, 1}, {11.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 1}, {});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test3) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});
  auto expected = NDArrayFactory::create<double>('c', {1, 1, 1}, {10.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 0, 1}, {});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test4) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});
  auto expected = NDArrayFactory::create<double>('c', {1, 1, 1}, {11.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 1, 1}, {});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test5) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});

  auto expected = NDArrayFactory::create<double>('c', {1, 1, 4}, {12., 7., 11., 10.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 0, 1}, {0, 1});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test6) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});

  auto expected = NDArrayFactory::create<double>('c', {1, 1, 4}, {16., 14., 15., 13.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 1, 1}, {0, 1});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test7) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});

  auto expected = NDArrayFactory::create<double>('c', {1, 1, 4}, {12., 7., 11., 10.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 1}, {0, 1});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test8) {
  const int dim0 = 5, dim1 = 5, dim2 = 4;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0, dim1, dim2},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});

  auto expected = NDArrayFactory::create<double>('c', {4}, {12., 7., 11., 10.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 0}, {0, 1});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test9) {
  const int dim0 = 100;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});

  auto expected = NDArrayFactory::create<double>(11.);

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 0}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test10) {
  const int dim0 = 100;

  auto input = NDArrayFactory::create<double>(
      'c', {dim0},
      {6.,  7.,  83., 81., 84., 86., 87.,  85., 88., 5.,  8.,  78., 79., 77., 80., 10., 16., 18., 19., 17.,
       20., 22., 23., 21., 24., 26., 27.,  25., 28., 30., 31., 29., 32., 38., 11., 9.,  12., 14., 15., 13.,
       39., 37., 40., 42., 43., 41., 44.,  46., 47., 45., 48., 50., 51., 49., 52., 54., 55., 53., 56., 58.,
       59., 57., 60., 98., 99., 97., 100., 62., 63., 61., 64., 66., 67., 65., 68., 70., 71., 69., 72., 74.,
       75., 73., 76., 2.,  3.,  1.,  4.,   94., 95., 93., 96., 82., 90., 91., 89., 92., 34., 35., 33., 36.});

  auto expected = NDArrayFactory::create<double>('c', {1}, {11.});

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 1}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test11) {
  const int dim0 = 1;

  auto input = NDArrayFactory::create<double>('c', {dim0}, {100.});

  auto expected = NDArrayFactory::create<double>('c', {1}, {100.});
  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 1}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, percentile_test12) {
  const int dim0 = 1;

  auto input = NDArrayFactory::create<double>('c', {dim0}, {100.});

  auto expected = NDArrayFactory::create<double>(100.);

  sd::ops::percentile op;
  // q,  interpolation, keepDims
  auto result = op.evaluate({&input}, {10, 2, 0}, {});
  auto output = result.at(0);

  ASSERT_TRUE(expected.isSameShape(output));
  ASSERT_TRUE(expected.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, transpose_test3) {
  auto input = NDArrayFactory::create<double>(
      'c', {5, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 5}, {1.f, 4.f, 7.f, 10.f, 13.f, 2.f, 5.f, 8.f, 11.f, 14.f, 3.f, 6.f, 9.f, 12.f, 15.f});

  sd::ops::transpose op;
  auto result = op.evaluate({&input}, {}, {});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, rationaltanh_test1) {
  auto input = NDArrayFactory::create<double>('c', {8}, {0, 1, 2, 3, 4, 5, 6, 7});
  NDArray exp =
      NDArrayFactory::create<double>({0.000000, 0.998222, 1.516093, 1.658054, 1.695077, 1.706884, 1.711427, 1.713446});

  sd::ops::rationaltanh op;
  auto result = op.evaluate({&input}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Output rationaltanh");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, rationaltanh_test2) {
  auto input = NDArrayFactory::create<double>('c', {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  NDArray exp = NDArrayFactory::create<double>(
      'c', {2, 2, 2}, {0.000000, 0.998222, 1.516093, 1.658054, 1.695077, 1.706884, 1.711427, 1.713446});

  sd::ops::rationaltanh op;
  auto result = op.evaluate({&input}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Output rationaltanh");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, rationaltanh_test3) {
  auto input = NDArrayFactory::create<double>('c', {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto eps = NDArrayFactory::create<double>('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  NDArray exp = NDArrayFactory::create<double>(
      'c', {2, 2, 2}, {1.143933, 1.605747, 0.795557, 0.261710, 0.095832, 0.041218, 0.020221, 0.010971});

  sd::ops::rationaltanh_bp op;
  auto result = op.evaluate({&input, &eps}, {}, {});
  auto output = result.at(0);
  //    output->printBuffer("Output rationaltanh BP");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, rectifiedtanh_test1) {
  auto input = NDArrayFactory::create<double>('c', {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  NDArray exp = NDArrayFactory::create<double>(
      'c', {2, 2, 2}, {0.000000, 0.761594, 0.964028, 0.995055, 0.999329, 0.999909, 0.999988, 0.999998});

  sd::ops::rectifiedtanh op;
  auto result = op.evaluate({&input}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Output rectifiedtanh");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, rectifiedtanh_test2) {
  auto input = NDArrayFactory::create<double>('c', {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto eps = NDArrayFactory::create<double>('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  NDArray exp = NDArrayFactory::create<double>(
      'c', {2, 2, 2}, {0.000000, 0.839949, 0.211952, 0.039464, 0.006705, 0.001089, 0.000172, 0.000027});

  sd::ops::rectifiedtanh_bp op;
  auto result = op.evaluate({&input, &eps}, {}, {});
  auto output = result.at(0);
  //    output->printBuffer("Output rectifiedtanh BP");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

TEST_F(DeclarableOpsTests7, RealDiv_1) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 2, 1}, {2.f, 4.f});
  NDArray y = NDArrayFactory::create<float>('c', {1, 2}, {1.f, 2.f});
  NDArray e = NDArrayFactory::create<float>('c', {1, 2, 2}, {2.f, 1.f, 4.f, 2.f});

  sd::ops::realdiv op;
  auto result = op.evaluate({&x, &y}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput RealDiv");
  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, RealDiv_BP_1) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 2, 1}, {2.f, 4.f});
  NDArray y = NDArrayFactory::create<float>('c', {1, 2}, {1.f, 2.f});
  NDArray e0 = NDArrayFactory::create<float>('c', {1, 2, 1}, {2.f, 5.f});
  NDArray e1 = NDArrayFactory::create<float>('c', {1, 2}, {-14.f, -5.f});
  NDArray eps = NDArrayFactory::create<float>('c', {1, 2, 2}, {1.f, 2.f, 3.f, 4.f});

  sd::ops::realdiv_bp op;
  auto result = op.evaluate({&x, &y, &eps}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z0 = result.at(0);
  auto z1 = result.at(1);
  //    z0->printShapeInfo("OUtput RealDiv BP0 shape");
  //    z1->printShapeInfo("OUtput RealDiv BP1 shape");
  //    z0->printIndexedBuffer("OUtput RealDiv BP0");
  //    z1->printIndexedBuffer("OUtput RealDiv BP1");
  //    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e0.equalsTo(z0));
  ASSERT_TRUE(e1.equalsTo(z1));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, ShapesOf_1) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 2, 1}, {2.f, 4.f});
  //    NDArray y = NDArrayFactory::create<float>('c', {1, 2}, {1,2});
  NDArray e = NDArrayFactory::create<sd::LongType>({1, 2, 1});

  sd::ops::shapes_of op;
  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput RealDiv");
  //    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, ShapesOf_2) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 2, 1}, {2.f, 4.f});
  NDArray y = NDArrayFactory::create<float>('c', {1, 2}, {1.f, 2.f});
  NDArray e0 = NDArrayFactory::create<sd::LongType>({1, 2, 1});
  NDArray e1 = NDArrayFactory::create<sd::LongType>({1, 2});

  sd::ops::shapes_of op;
  auto result = op.evaluate({&x, &y}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z0 = result.at(0);
  auto z1 = result.at(1);
  //    z0->printIndexedBuffer("OUtput shapes2");
  //    z1->printIndexedBuffer("OUtput shapes2");
  //    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e0.equalsTo(z0));
  ASSERT_TRUE(e1.equalsTo(z1));
}

TEST_F(DeclarableOpsTests7, Size_1) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 2, 1}, {2.f, 4.f});
  NDArray y = NDArrayFactory::create<float>('c', {5, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 7.f, 9.f, 10.f, 10.f, 11.f});
  NDArray e = NDArrayFactory::create<sd::LongType>(2);

  sd::ops::size op;
  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput SIZE");
  ///    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(DeclarableOpsTests7, Size_2) {
  NDArray x = NDArrayFactory::create<double>('c', {1, 2, 1}, {2, 4});
  NDArray y = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray e = NDArrayFactory::create<sd::LongType>(10);

  sd::ops::size op;
  auto result = op.evaluate({&y}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput SIZE");
  ///    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(DeclarableOpsTests7, Softplus_1) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray e = NDArrayFactory::create<double>(
      'c', {5, 2},
      {1.3132616, 2.126928, 3.0485873, 4.01815, 5.0067153, 7.0009117, 9.000123, 10.000046, 10.000046, 11.000016});

  sd::ops::softplus op;
  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput Softplus");
  ///    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(DeclarableOpsTests7, Softplus_BP_1) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  //    NDArray e = NDArrayFactory::create<float>('c', {5, 2},
  //    {1.3132616,  2.126928, 3.0485873, 4.01815, 5.0067153, 7.0009117, 9.000123, 10.000046, 10.000046, 11.000016});
  NDArray eps = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  sd::ops::softplus ffOP;
  sd::ops::softplus_bp bpOp;
  const OpArgsHolder argsHolderFF({&x}, {}, {});
  const OpArgsHolder argsHolderBP({&x, &eps}, {}, {});

  bool gradOK = GradCheck::checkGrad(ffOP, bpOp, argsHolderFF, argsHolderBP);

  ASSERT_TRUE(gradOK);
  //
  //    auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput Softplus");
  /////    ASSERT_TRUE(e.isSameShape(z));
  //    ASSERT_TRUE(e.equalsTo(*z));
  //
  //
}

TEST_F(DeclarableOpsTests7, Softsign_1) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray e = NDArrayFactory::create<double>(
      'c', {5, 2}, {0.5, 0.6666667, 0.75, 0.8, 0.8333333, 0.875, 0.9, 0.90909094, 0.90909094, 0.9166667});

  sd::ops::softsign op;
  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);
  //    z->printIndexedBuffer("OUtput Softsign");
  ///    ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(DeclarableOpsTests7, Softsign_BP_1) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  //    NDArray e = NDArrayFactory::create<float>('c', {5, 2},
  //    {1.3132616f,  2.126928f, 3.0485873f, 4.01815f, 5.0067153f, 7.0009117f, 9.000123f, 10.000046f, 10.000046f, 11.000016f});
  NDArray eps = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  sd::ops::softsign ffOP;
  sd::ops::softsign_bp bpOp;
  const OpArgsHolder argsHolderFF({&x}, {}, {});
  const OpArgsHolder argsHolderBP({&x, &eps}, {}, {});

  bool gradOK = GradCheck::checkGrad(ffOP, bpOp, argsHolderFF, argsHolderBP);

  ASSERT_TRUE(gradOK);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, fill_test2) {
  auto x = NDArrayFactory::create<int>('c', {1, 2}, {2, 2});
  auto v = NDArrayFactory::create<double>(42.);
  auto exp = NDArrayFactory::create<double>('c', {2, 2}, {42.f, 42.f, 42.f, 42.f});

  sd::ops::fill op;
  auto result = op.evaluate({&x, &v}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, fill_test3) {
  auto x = NDArrayFactory::create<int>('c', {2}, {2, 2});
  auto v = NDArrayFactory::create<double>(42.);
  auto exp = NDArrayFactory::create<double>('c', {2, 2}, {42.f, 42.f, 42.f, 42.f});

  sd::ops::fill op;
  auto result = op.evaluate({&x, &v}, {}, {});
  auto output = result.at(0);

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, ToggleBits_test1) {
  auto x = NDArrayFactory::create<int>('c', {2}, {2, 2});
  auto exp = NDArrayFactory::create<int>('c', {2}, {-3, -3});

  sd::ops::toggle_bits op;
  auto result = op.evaluate({&x});
  auto output = result.at(0);

  ASSERT_EQ(sd::Status::OK, result.status());
  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, ToggleBits_test2) {
  auto x = NDArrayFactory::create<int>('c', {2}, {2, 2});
  auto y = NDArrayFactory::create<int>('c', {2}, {1, 1});
  auto exp0 = NDArrayFactory::create<int>('c', {2}, {-3, -3});
  auto exp1 = NDArrayFactory::create<int>('c', {2}, {-2, -2});

  sd::ops::toggle_bits op;
  auto result = op.evaluate({&x, &y});
  auto output = result.at(0);
  auto z = result.at(1);

  ASSERT_EQ(sd::Status::OK, result.status());
  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(exp0.isSameShape(output));
  ASSERT_TRUE(exp0.equalsTo(output));
  ASSERT_TRUE(exp1.isSameShape(z));
  ASSERT_TRUE(exp1.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Truncatediv_test1) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray y = NDArrayFactory::create<double>('c', {5, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
  NDArray exp = NDArrayFactory::create<double>('c', {5, 2}, {0.5, 1., 1.5, 2., 2.5, 3.5, 4.5, 5., 5., 5.5});

  sd::ops::truncatediv op;
  auto result = op.evaluate({&x, &y}, {}, {});
  ASSERT_EQ(sd::Status::OK, result.status());
  auto output = result.at(0);
  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(exp.isSameShape(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Truncatediv_test2) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray y = NDArrayFactory::create<double>('c', {1, 2}, {2, 2});
  NDArray exp = NDArrayFactory::create<double>('c', {5, 2}, {0.5, 1., 1.5, 2., 2.5, 3.5, 4.5, 5., 5., 5.5});

  sd::ops::truncatediv op;
  auto result = op.evaluate({&x, &y}, {}, {});
  ASSERT_EQ(sd::Status::OK, result.status());
  auto output = result.at(0);
  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(exp.isSameShape(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TypesConversion_test1) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray expI = NDArrayFactory::create<int>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray expL = NDArrayFactory::create<sd::LongType>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray expF = NDArrayFactory::create<float>('c', {5, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 7.f, 9.f, 10.f, 10.f, 11.f});
  NDArray expF16 = NDArrayFactory::create<float16>('c', {5, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 7.f, 9.f, 10.f, 10.f, 11.f});

  sd::ops::to_int32 op32;
  sd::ops::to_int64 op64;
  auto result32 = op32.evaluate({&x}, {}, {});
  auto result64 = op64.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result32.status());
  ASSERT_EQ(sd::Status::OK, result64.status());
  auto out1 = result32.at(0);
  //    out1->printIndexedBuffer("OUT_I");
  auto out2 = result64.at(0);
  //    out2->printIndexedBuffer("OUT_L");

  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(expI.equalsTo(out1));
  ASSERT_TRUE(expL.equalsTo(out2));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TypesConversion_test2) {
  NDArray x = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray expF = NDArrayFactory::create<float>('c', {5, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 7.f, 9.f, 10.f, 10.f, 11.f});
  NDArray expH = NDArrayFactory::create<float16>('c', {5, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 7.f, 9.f, 10.f, 10.f, 11.f});

  sd::ops::to_float32 op32;
  sd::ops::to_float16 op16;
  auto result32 = op32.evaluate({&x}, {}, {});
  auto result16 = op16.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result32.status());
  ASSERT_EQ(sd::Status::OK, result16.status());
  auto out1 = result32.at(0);
  //    out1->printIndexedBuffer("OUT_F");
  auto out2 = result16.at(0);
  //    out2->printIndexedBuffer("OUT_H");

  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(expF.equalsTo(out1));
  ASSERT_TRUE(expH.equalsTo(out2));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TypesConversion_test3) {
  NDArray x = NDArrayFactory::create<sd::LongType>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray exp32 = NDArrayFactory::create<unsigned int>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray exp64 = NDArrayFactory::create<uint64_t>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});

  sd::ops::to_uint32 op32;
  sd::ops::to_uint64 op64;
  auto result32 = op32.evaluate({&x}, {}, {});
  auto result64 = op64.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result32.status());
  ASSERT_EQ(sd::Status::OK, result64.status());
  auto out1 = result32.at(0);
  //    out1->printIndexedBuffer("OUT_U32");
  auto out2 = result64.at(0);
  //    out2->printIndexedBuffer("OUT_U64");

  //    output->printIndexedBuffer("Toggled");
  ASSERT_TRUE(exp32.equalsTo(out1));
  ASSERT_TRUE(exp64.equalsTo(out2));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, TypesConversion_test4) {
  NDArray x = NDArrayFactory::create<sd::LongType>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});
  NDArray exp32 = NDArrayFactory::create<float>('c', {5, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 7.f, 9.f, 10.f, 10.f, 11.f});
  NDArray exp64 = NDArrayFactory::create<double>('c', {5, 2}, {1, 2, 3, 4, 5, 7, 9, 10, 10, 11});

  sd::ops::to_float32 op32;
  sd::ops::to_double op64;
  auto result32 = op32.evaluate({&x}, {}, {});
  auto result64 = op64.evaluate({&x}, {}, {});

  ASSERT_EQ(sd::Status::OK, result32.status());
  ASSERT_EQ(sd::Status::OK, result64.status());
  auto out1 = result32.at(0);
  auto out2 = result64.at(0);

  ASSERT_TRUE(exp32.equalsTo(out1));
  ASSERT_TRUE(exp64.equalsTo(out2));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test1) {
  auto input = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {1, 1, 2, 2});

  auto exp = NDArrayFactory::create<double>(
      'c', {4, 7}, {2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test2) {
  auto input = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {1, 1, 2, 2});

  auto exp = NDArrayFactory::create<double>(
      'c', {4, 7}, {6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test3) {
  auto input = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {1, 2}, {2, 2});

  auto exp = NDArrayFactory::create<double>('c', {7}, {2, 1, 1, 2, 3, 3, 2});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test4) {
  auto input = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {2}, {2, 3});

  auto exp = NDArrayFactory::create<double>('c', {8}, {2, 1, 1, 2, 3, 3, 2, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test5) {
  auto input = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {2}, {2, 2});

  auto exp = NDArrayFactory::create<double>('c', {7}, {3, 2, 1, 2, 3, 2, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  auto output = result.at(0);
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test6) {
  auto input = NDArrayFactory::create<double>(1.);
  auto paddings = NDArrayFactory::create<int>('c', {1, 2, 1, 1}, {1, 1});

  auto exp = NDArrayFactory::create<double>('c', {3}, {1, 1, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test7) {
  auto input = NDArrayFactory::create<double>(1.);
  auto paddings = NDArrayFactory::create<int>('c', {2}, {1, 1});

  auto exp = NDArrayFactory::create<double>('c', {3}, {1, 1, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test8) {
  auto input = NDArrayFactory::create<double>('c', {1, 3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {1, 1, 3, 3});

  auto exp = NDArrayFactory::create<double>(
      'c', {3, 9}, {3, 2, 1, 1, 2, 3, 3, 2, 1, 3, 2, 1, 1, 2, 3, 3, 2, 1, 3, 2, 1, 1, 2, 3, 3, 2, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  ASSERT_EQ(result.status(), sd::Status::OK);

  auto output = result.at(0);
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test9) {
  auto input = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {2, 2, 3, 3});

  auto exp = NDArrayFactory::create<double>(
      'c', {6, 9}, {6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1, 3, 2, 1, 1, 2, 3, 3, 2, 1,
                    6, 5, 4, 4, 5, 6, 6, 5, 4, 6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test10) {
  auto input = NDArrayFactory::create<double>('c', {1, 3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

  auto exp = NDArrayFactory::create<double>('c', {1, 3}, {1., 2., 3.});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test11) {
  auto input = NDArrayFactory::create<double>('c', {1, 3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

  auto exp = NDArrayFactory::create<double>('c', {1, 3}, {1., 2., 3.});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test12) {
  auto input = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 1}, {0, 0});

  auto exp = NDArrayFactory::create<double>('c', {3}, {1., 2., 3.});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test13) {
  auto input = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

  auto exp = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test14) {
  auto input = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {1LL, 0LL, 0LL, 1LL});

  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {4, 5, 6, 5, 1, 2, 3, 2, 4, 5, 6, 5});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test15) {
  auto input = NDArrayFactory::create<double>('c', {2, 3}, {1., 2., 3., 4., 5., 6.});
  auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {1, 1, 0, 0});

  auto exp = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6});

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {1});
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, mirrorPad_test16) {
  auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
  auto paddings = NDArrayFactory::create<int>('c', {3, 2}, {3, 3, 2, 2, 1, 1});

  auto exp = NDArrayFactory::create<double>(
      'c', {10, 7, 4},
      {24., 23., 24., 23., 22., 21., 22., 21., 20., 19., 20., 19., 22., 21., 22., 21., 24., 23., 24., 23., 22., 21.,
       22., 21., 20., 19., 20., 19., 18., 17., 18., 17., 16., 15., 16., 15., 14., 13., 14., 13., 16., 15., 16., 15.,
       18., 17., 18., 17., 16., 15., 16., 15., 14., 13., 14., 13., 12., 11., 12., 11., 10., 9.,  10., 9.,  8.,  7.,
       8.,  7.,  10., 9.,  10., 9.,  12., 11., 12., 11., 10., 9.,  10., 9.,  8.,  7.,  8.,  7.,  6.,  5.,  6.,  5.,
       4.,  3.,  4.,  3.,  2.,  1.,  2.,  1.,  4.,  3.,  4.,  3.,  6.,  5.,  6.,  5.,  4.,  3.,  4.,  3.,  2.,  1.,
       2.,  1.,  12., 11., 12., 11., 10., 9.,  10., 9.,  8.,  7.,  8.,  7.,  10., 9.,  10., 9.,  12., 11., 12., 11.,
       10., 9.,  10., 9.,  8.,  7.,  8.,  7.,  18., 17., 18., 17., 16., 15., 16., 15., 14., 13., 14., 13., 16., 15.,
       16., 15., 18., 17., 18., 17., 16., 15., 16., 15., 14., 13., 14., 13., 24., 23., 24., 23., 22., 21., 22., 21.,
       20., 19., 20., 19., 22., 21., 22., 21., 24., 23., 24., 23., 22., 21., 22., 21., 20., 19., 20., 19., 18., 17.,
       18., 17., 16., 15., 16., 15., 14., 13., 14., 13., 16., 15., 16., 15., 18., 17., 18., 17., 16., 15., 16., 15.,
       14., 13., 14., 13., 12., 11., 12., 11., 10., 9.,  10., 9.,  8.,  7.,  8.,  7.,  10., 9.,  10., 9.,  12., 11.,
       12., 11., 10., 9.,  10., 9.,  8.,  7.,  8.,  7.,  6.,  5.,  6.,  5.,  4.,  3.,  4.,  3.,  2.,  1.,  2.,  1.,
       4.,  3.,  4.,  3.,  6.,  5.,  6.,  5.,  4.,  3.,  4.,  3.,  2.,  1.,  2.,  1.});
  input.linspace(1.);

  sd::ops::mirror_pad op;
  auto result = op.evaluate({&input, &paddings}, {}, {0});
  ASSERT_EQ(result.status(), sd::Status::OK);
  auto output = result.at(0);
  // output->printBuffer("VVV");
  // exp.printBuffer("EXP");

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_1) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 5}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
  auto exp = NDArrayFactory::create<double>(120.f);
  //************************************//

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&input}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  // z->printIndexedBuffer("Result is ");
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_2) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 5}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
  auto exp = NDArrayFactory::create<double>({15.f, 40.f, 65.f});
  //************************************//

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&input}, {}, {1});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_1) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 5}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
  auto exp = NDArrayFactory::create<double>(1307674368000.f);
  //************************************//

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&input}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  // z->printIndexedBuffer("Result is ");
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_2) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 5}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
  auto exp = NDArrayFactory::create<double>({120.f, 30240.f, 360360.f});
  //************************************//

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&input}, {}, {1});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_01) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
  x.linspace(1);

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {66.f, 72.f, 78.f, 84.f});
  x.linspace(1);

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {1.}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
  x.linspace(1);

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {68.f, 100.f, 132.f});
  x.linspace(1);

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {1.}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(300.f);
  x.linspace(1);

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(300.f);
  x.linspace(1);

  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {300.f});
  x.linspace(1);
  //    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
  sd::ops::reduce_sum op;
  auto result = op.evaluate({&x}, {1.}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_01) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>('c', {2}, {10395.f, 46080.f});
  x.linspace(1);

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 2}, {10395.f, 46080.f});
  x.linspace(1);

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {1.}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>('c', {3}, {112.f, 1080.f, 3960.f});
  x.linspace(1);

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {112.f, 1080.f, 3960.f});
  x.linspace(1);

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {1.}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>(479001600.f);
  x.linspace(1);

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>(479001600.f);
  x.linspace(1);

  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 2});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {479001600.f});
  x.linspace(1);
  //    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
  sd::ops::reduce_prod op;
  auto result = op.evaluate({&x}, {1.}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

TYPED_TEST(TypedDeclarableOpsTests7, Test_Pnorm_Once_Again) {
  auto input = NDArrayFactory::create<TypeParam>(
      'c', {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f,
                          14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
  auto exp = NDArrayFactory::create<TypeParam>(
      'c', {1, 1, 5, 5}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f,
                          14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});

  sd::ops::pnormpool2d op;
  auto result = op.evaluate({&input}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 1, 3, 0});
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_EQ(exp, *result.at(0));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  x.linspace(1);

  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
  x.linspace(1);

  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {1.}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {1.f, 5.f, 9.f});
  x.linspace(1);

  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {1.f, 5.f, 9.f});
  x.linspace(1);

  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {1.}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(1.f);
  x.linspace(1);

  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(1.f);
  x.linspace(1);

  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {1.f});
  x.linspace(1);
  //    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
  sd::ops::reduce_min op;
  auto result = op.evaluate({&x}, {1.}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  //    output->printShapeInfo("Output shape");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {21.f, 22.f, 23.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {1.}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {1.}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(24.f);
  x.linspace(1);

  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(24.f);
  x.linspace(1);

  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
  x.linspace(1);
  //    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
  sd::ops::reduce_max op;
  auto result = op.evaluate({&x}, {1.}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {66.f, 72.f, 78.f, 84.f});
  x.linspace(1);

  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {66.f, 72.f, 78.f, 84.f});
  x.linspace(1);

  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {1.}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {68.f, 100.f, 132.f});
  x.linspace(1);

  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {68.f, 100.f, 132.f});
  x.linspace(1);

  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {1.}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(300.f);
  x.linspace(1);

  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(300.f);
  x.linspace(1);

  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {300.f});
  x.linspace(1);
  //    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
  sd::ops::reduce_norm1 op;
  auto result = op.evaluate({&x}, {1.}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
  x.linspace(1);

  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
  x.linspace(1);

  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {1.}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {29.597298f, 39.344631f, 49.759422f});
  x.linspace(1);

  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {29.597298f, 39.344631f, 49.759422f});
  x.linspace(1);

  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {1.}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(70.f);
  x.linspace(1);

  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(70.f);
  x.linspace(1);

  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {70.f});
  x.linspace(1);
  //    x.printIndexedBuffer("Input with shape (2, 3, 4) is");
  sd::ops::reduce_norm2 op;
  auto result = op.evaluate({&x}, {1.}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {21.f, 22.f, 23.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {1.f}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {16.f, 20.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {16.f, 20.f, 24.f});
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {1.f}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(24.f);
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(24.f);
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {24.f});
  x.linspace(1);

  sd::ops::reduce_norm_max op;
  auto result = op.evaluate({&x}, {1.f}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {4}, {1006.f, 1144.f, 1294.f, 1456.f});
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 4}, {1006.f, 1144.f, 1294.f, 1456.f});
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {1.f}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3}, {876.f, 1548.f, 2476.f});
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 3, 1}, {876.f, 1548.f, 2476.f});
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {1.f}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(4900.f);
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>(4900.f);
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto exp = NDArrayFactory::create<double>('c', {1, 1, 1}, {4900.f});
  x.linspace(1);

  sd::ops::reduce_sqnorm op;
  auto result = op.evaluate({&x}, {1.f}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_BP_1) {
  auto input = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  auto eps = NDArrayFactory::create<double>(0.5f);
  auto exp = NDArrayFactory::create<double>('c', {3, 4},
                                            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  //************************************//

  sd::ops::reduce_sum_bp op;
  auto result = op.evaluate({&input, &eps}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_BP_2) {
  auto input = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  auto eps = NDArrayFactory::create<double>('c', {1, 1}, {0.5f});
  auto exp = NDArrayFactory::create<double>('c', {3, 4},
                                            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  //************************************//

  sd::ops::reduce_sum_bp op;
  auto result = op.evaluate({&input, &eps}, {1.f}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //  z->printIndexedBuffer("Result is ");
  //  z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_BP_3) {
  auto input = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  //************************************//

  sd::ops::reduce_sum_bp op;
  auto result = op.evaluate({&input, &eps}, {}, {0});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Sum_BP_4) {
  auto input = NDArrayFactory::create<double>('c', {3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  //************************************//

  sd::ops::reduce_sum_bp op;
  auto result = op.evaluate({&input, &eps}, {1.f}, {0});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_BP_1) {
  auto input = NDArrayFactory::create<double>(
      'c', {3, 5}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
  auto eps = NDArrayFactory::create<double>(1307674368000.f);
  //************************************//
  //    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
  //    0.5f, 0.5f,0.5f});
  //************************************//
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 5},
      {1710012166826558903812096.f, 855006083413279451906048.f, 570004067618451974258688.f, 427503041706639725953024.f,
       342002454982589992140800.f, 285002033809225987129344.f, 244287457550765131825152.f, 213751520853319862976512.f,
       190001355872817324752896.f, 171001227491294996070400.f, 155455648254341989531648.f, 142501016904612993564672.f,
       131539399526781282156544.f, 122143728775382565912576.f, 114000815325130245799936.f});

  sd::ops::reduce_prod_bp op;
  auto result = op.evaluate({&input, &eps}, {}, {});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_BP_2) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  auto eps = NDArrayFactory::create<double>(0.5f);
  //************************************//
  //    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
  //    0.5f, 0.5f,0.5f});
  //************************************//
  auto exp = NDArrayFactory::create<double>('c', {3, 4});

  sd::ops::reduce_prod_bp op;
  sd::ops::reduce_prod op_exp;
  auto res = op_exp.evaluate({&input});
  auto result = op.evaluate({&input, &eps}, {}, {});
  exp.assign(res.at(0)->e<double>(0));
  exp /= input;
  exp *= eps.e<double>(0);
  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  // z->printIndexedBuffer("Result is ");
  // exp.printIndexedBuffer("Expected");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_BP_3) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
  //************************************//
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4}, {45.f, 120.f, 231.f, 384.f, 9.f, 40.f, 99.f, 192.f, 5.f, 24.f, 63.f, 128.f});

  sd::ops::reduce_prod_bp op;
  // sd::ops::reduce_prod op_exp;
  auto result = op.evaluate({&input, &eps}, {1.f}, {0});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    exp.printIndexedBuffer("Expected");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_BP_03) {
  int ax = 0;
  auto input =
      NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
  //************************************//
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4}, {45.f, 120.f, 231.f, 384.f, 9.f, 40.f, 99.f, 192.f, 5.f, 24.f, 63.f, 128.f});
  auto axis = NDArrayFactory::create<int>('c', {1}, {ax});
  sd::ops::reduce_prod_bp op;
  // sd::ops::reduce_prod op_exp;
  auto result = op.evaluate({&input, &eps, &axis}, {}, {}, {true});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    exp.printIndexedBuffer("Expected");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_BP_4) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  //************************************//
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4}, {45.f, 120.f, 231.f, 384.f, 9.f, 40.f, 99.f, 192.f, 5.f, 24.f, 63.f, 128.f});

  sd::ops::reduce_prod_bp op;
  sd::ops::reduce_prod op_exp;
  //    auto res = op_exp.execute({&input}, {}, {});
  auto result = op.evaluate({&input, &eps}, {0.f}, {0});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    exp.printIndexedBuffer("Expected");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));

  //
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Prod_BP_5) {
  auto input =
      NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  auto eps = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
  //************************************//
  auto exp = NDArrayFactory::create<double>(
      'c', {3, 4}, {24.f, 12.f, 8.f, 6.f, 672.f, 560.f, 480.f, 420.f, 3960.f, 3564.f, 3240.f, 2970.f});

  sd::ops::reduce_prod_bp op;
  sd::ops::reduce_prod op_exp;
  //    auto res = op_exp.execute({&input}, {}, {});
  auto result = op.evaluate({&input, &eps}, {0.f}, {1});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto z = result.at(0);
  //    z->printIndexedBuffer("Result is ");
  //    exp.printIndexedBuffer("Expected");
  //    z->printShapeInfo();
  ASSERT_TRUE(exp.equalsTo(z));

  //
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  exp.p(0, eps.e<double>(0));
  exp.p(1, eps.e<double>(1));
  exp.p(2, eps.e<double>(2));
  exp.p(3, eps.e<double>(3));
  x.linspace(1);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  exp.p(0, eps.e<double>(0));
  exp.p(1, eps.e<double>(1));
  exp.p(2, eps.e<double>(2));
  exp.p(3, eps.e<double>(3));
  x.linspace(1);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  exp.p(0, eps.e<double>(0));
  exp.p(1, eps.e<double>(1));
  exp.p(2, eps.e<double>(2));
  exp.p(3, eps.e<double>(3));
  auto axes = NDArrayFactory::create<int>({0, 1});
  x.linspace(1);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps, &axes}, {}, {}, {true});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_3) {
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1}, {0.5f});
  auto exp = NDArrayFactory::create<double>('c', {3, 4});
  x.linspace(1);
  x.p(2, 2, -1.f);
  exp.p(2, 2, 0.5f);
  // x.printIndexedBuffer("Input is");
  // exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {});
  auto output = result.at(0);
  // output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_4) {
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto eps = NDArrayFactory::create<double>(0.5f);
  auto exp = NDArrayFactory::create<double>('c', {3, 4});
  x.linspace(1);
  x.p(2, 2, -1.f);
  exp.p(2, 2, 0.5f);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_5) {
  auto x = NDArrayFactory::create<double>('c', {4, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {4, 4});
  x.linspace(1);
  x.p(0, 0, -1.f);
  x.p(1, 1, -2.f);
  x.p(2, 2, -3.f);
  x.p(3, 3, -4.f);
  exp.p(0, 0, 1.f);
  exp.p(1, 1, 2.f);
  exp.p(2, 2, 3.f);
  exp.p(3, 3, 4.f);
  //    exp(2,2) = 0.5f;
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Min_BP_6) {
  auto x = NDArrayFactory::create<double>('c', {4, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {4, 4});
  x.linspace(1);
  x.p(0, 0, -1.f);
  x.p(1, 1, -2.f);
  x.p(2, 2, -3.f);
  x.p(3, 3, -4.f);
  exp.p(0, 0, 1.f);
  exp.p(1, 1, 2.f);
  exp.p(2, 2, 3.f);
  exp.p(3, 3, 4.f);
  //    exp(2,2) = 0.5f;
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_min_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {21.f, 22.f, 23.f, 24.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  exp.p(20, eps.e<double>(0));
  exp.p(21, eps.e<double>(1));
  exp.p(22, eps.e<double>(2));
  exp.p(23, eps.e<double>(3));
  x.linspace(1);
  // x.printIndexedBuffer("Input is");
  // exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_max_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1});
  auto output = result.at(0);
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_BP_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {21.f, 22.f, 23.f, 24.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  exp.p(20, eps.e<double>(0));
  exp.p(21, eps.e<double>(1));
  exp.p(22, eps.e<double>(2));
  exp.p(23, eps.e<double>(3));
  x.linspace(1);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_max_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_BP_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {21.f, 22.f, 23.f, 24.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  exp.p(20, eps.e<double>(0));
  exp.p(21, eps.e<double>(1));
  exp.p(22, eps.e<double>(2));
  exp.p(23, eps.e<double>(3));
  auto axes = NDArrayFactory::create<int>({0, 1});
  x.linspace(1);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_max_bp op;
  auto result = op.evaluate({&x, &eps, &axes}, {}, {}, {true});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_BP_3) {
  auto x = NDArrayFactory::create<double>('c', {4, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {4, 4});
  x.linspace(1);
  x.p(0, 0, 21.f);
  x.p(1, 1, 22.f);
  x.p(2, 2, 23.f);
  x.p(3, 3, 24.f);
  exp.p(0, 0, 1.f);
  exp.p(1, 1, 2.f);
  exp.p(2, 2, 3.f);
  exp.p(3, 3, 4.f);
  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_max_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Max_BP_4) {
  auto x = NDArrayFactory::create<double>('c', {4, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {4, 4});
  x.linspace(1);
  x.p(0, 0, 21.f);
  x.p(1, 1, 22.f);
  x.p(2, 2, 23.f);
  x.p(3, 3, 24.f);
  exp.p(0, 0, 1.f);
  exp.p(1, 1, 2.f);
  exp.p(2, 2, 3.f);
  exp.p(3, 3, 4.f);

  //    x.printIndexedBuffer("Input is");
  //    exp.printIndexedBuffer("Expected ");
  sd::ops::reduce_max_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>(5.f);
  x.linspace(1);
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.p(12, -2.f);
  x.p(20, -3.f);
  exp.assign(5.f);
  exp.p(12, -exp.e<double>(12));
  exp.p(20, -exp.e<double>(20));
  sd::ops::reduce_norm1_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_BP_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>({1.f, 2.f, 3.f, 4.f});
  x.linspace(1);
  auto exp =
      NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                                      1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  sd::ops::reduce_norm1_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1});
  ASSERT_EQ(sd::Status::OK, result.status());
  auto output = result.at(0);
  // output->printIndexedBuffer("Result is");
  // exp.printIndexedBuffer("Expect is");
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_BP_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>({1.f, 2.f, 3.f, 4.f});
  x.linspace(1);
  auto exp =
      NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                                      1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  auto axes = NDArrayFactory::create<int>({0, 1});
  sd::ops::reduce_norm1_bp op;
  auto result = op.evaluate({&x, &eps, &axes}, {}, {}, {false});
  ASSERT_EQ(sd::Status::OK, result.status());
  auto output = result.at(0);
  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm1_BP_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
  x.linspace(1);
  auto exp =
      NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                                      1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  sd::ops::reduce_norm1_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 1});
  auto output = result.at(0);
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
  x.linspace(1);

  sd::ops::reduce_norm2_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1});
  auto output = result.at(0);
  // output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(x.isSameShape(output));
  ASSERT_TRUE(x.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_BP_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
  x.linspace(1);

  sd::ops::reduce_norm2_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(x.isSameShape(output));
  ASSERT_TRUE(x.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_BP_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {31.7175f, 33.823071f, 35.97221f, 38.15757f});
  auto axes = NDArrayFactory::create<int>({0, 1});
  x.linspace(1);

  sd::ops::reduce_norm2_bp op;
  auto result = op.evaluate({&x, &eps, &axes}, {}, {}, {true});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(x.isSameShape(output));
  ASSERT_TRUE(x.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_BP_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {3}, {29.597298f, 39.344631f, 49.759422f});
  x.linspace(1);

  sd::ops::reduce_norm2_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(x.isSameShape(output));
  ASSERT_TRUE(x.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Norm2_BP_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 3, 1}, {29.597298f, 39.344631f, 49.759422f});
  x.linspace(1);

  sd::ops::reduce_norm2_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(x.isSameShape(output));
  ASSERT_TRUE(x.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 4}, {2.f,  8.f,  18.f, 32.f,  10.f, 24.f, 42.f,  64.f,  18.f, 40.f, 66.f,  96.f,
                       26.f, 56.f, 90.f, 128.f, 34.f, 72.f, 114.f, 160.f, 42.f, 88.f, 138.f, 192.f});
  x.linspace(1);

  sd::ops::reduce_sqnorm_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_SquaredNorm_BP_01) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>(
      'c', {2, 3, 4}, {2.f,  8.f,  18.f, 32.f,  10.f, 24.f, 42.f,  64.f,  18.f, 40.f, 66.f,  96.f,
                       26.f, 56.f, 90.f, 128.f, 34.f, 72.f, 114.f, 160.f, 42.f, 88.f, 138.f, 192.f});
  auto axes = NDArrayFactory::create<int>({0, 1});
  x.linspace(1);

  sd::ops::reduce_sqnorm_bp op;
  auto result = op.evaluate({&x, &eps, &axes}, {}, {}, {false});

  ASSERT_EQ(sd::Status::OK, result.status());
  auto output = result.at(0);

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  exp.p(20, 1.f);
  exp.p(21, 2.f);
  exp.p(22, 3.f);
  exp.p(23, 4.f);

  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  exp.p(20, 1.f);
  exp.p(21, 2.f);
  exp.p(22, 3.f);
  exp.p(23, 4.f);

  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 1});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 4}, {1.f, 2.f, 3.f, 4.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto axes = NDArrayFactory::create<int>({0, 1});
  x.linspace(1);
  exp.p(20, 1.f);
  exp.p(21, 2.f);
  exp.p(22, 3.f);
  exp.p(23, 4.f);

  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps, &axes}, {}, {}, {true});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_3) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);

  exp.p(15, 1.f);
  exp.p(19, 2.f);
  exp.p(23, 3.f);

  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_4) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 3, 1}, {1.f, 2.f, 3.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  exp.p(15, 1.f);
  exp.p(19, 2.f);
  exp.p(23, 3.f);
  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {0, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_5) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>(1.f);
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  exp.p(23, 1.f);
  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_6) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>(1.f);
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  exp.p(23, 1.f);

  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 1, 2});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_NormMax_BP_7) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto eps = NDArrayFactory::create<double>('c', {1, 1, 1}, {1.f});
  auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  exp.p(23, 1.f);
  sd::ops::reduce_norm_max_bp op;
  auto result = op.evaluate({&x, &eps}, {1.f}, {});
  auto output = result.at(0);
  //    output->printIndexedBuffer("Result is");

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Dot_BP_1) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto y = NDArrayFactory::create<double>('c', {2, 3, 4});
  NDArray* z;  // = NDArrayFactory::create<double>('c', {4});
  auto eps = NDArrayFactory::create<double>(1.f);
  //    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4});
  x.linspace(1);
  y.linspace(2);

  sd::ops::reduce_dot_bp op;
  auto result = op.evaluate({&x, &y, &eps}, {}, {});
  auto output = result.at(0);
  auto outputX = result.at(1);
  // tput->printIndexedBuffer("Result is");

  //    ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(x.equalsTo(outputX));
  ASSERT_TRUE(y.equalsTo(output));

  //    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Dot_BP_2) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto y = NDArrayFactory::create<double>('c', {2, 3, 4});
  //    auto z; // = NDArrayFactory::create<double>('c', {4});
  auto eps = NDArrayFactory::create<double>('c', {2, 4});
  auto expX = NDArrayFactory::create<double>('c', {2, 3, 4},
                                             {2.f,  4.f,  6.f,  8.f,  2.f,  4.f,  6.f,  8.f,  2.f,  4.f,  6.f,  8.f,
                                              10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f});
  auto expY =
      NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                                      5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f});
  x.assign(1.f);
  eps.linspace(1);
  y.assign(2.f);
  sd::ops::reduce_dot_bp op;
  auto result = op.evaluate({&x, &y, &eps}, {}, {1});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  auto outputX = result.at(0);
  auto outputY = result.at(1);
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(expX.equalsTo(outputX));
  ASSERT_TRUE(expY.equalsTo(outputY));

  //    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Dot_BP_02) {
  auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
  auto y = NDArrayFactory::create<double>('c', {2, 3, 4});
  //    auto z; // = NDArrayFactory::create<double>('c', {4});
  auto eps = NDArrayFactory::create<double>('c', {2, 4});
  auto expX = NDArrayFactory::create<double>('c', {2, 3, 4},
                                             {2.f,  4.f,  6.f,  8.f,  2.f,  4.f,  6.f,  8.f,  2.f,  4.f,  6.f,  8.f,
                                              10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f, 10.f, 12.f, 14.f, 16.f});
  auto expY =
      NDArrayFactory::create<double>('c', {2, 3, 4}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f,
                                                      5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f, 5.f, 6.f, 7.f, 8.f});
  auto axis = NDArrayFactory::create<int>('c', {1}, {1});
  x.assign(1.f);
  eps.linspace(1);
  y.assign(2.f);
  sd::ops::reduce_dot_bp op;
  auto result = op.evaluate({&x, &y, &eps, &axis}, {}, {}, {false});
  ASSERT_EQ(result.status(), sd::Status::OK);
  ASSERT_EQ(result.size(), 2);
  auto outputX = result.at(0);
  auto outputY = result.at(1);
  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(expX.equalsTo(outputX));
  ASSERT_TRUE(expY.equalsTo(outputY));

  //    delete z;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, Test_Reduce_Dot_BP_3) {
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto y = NDArrayFactory::create<double>('c', {3, 4});
  auto eps = NDArrayFactory::create<double>('c', {3});
  auto expX = NDArrayFactory::create<double>('c', {3, 4}, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f});
  auto expY =
      NDArrayFactory::create<double>('c', {3, 4}, {1.f, 2.f, 3.f, 4.f, 10.f, 12.f, 14.f, 16.f, 27.f, 30.f, 33.f, 36.f});
  x.linspace(1);
  eps.linspace(1);
  y.assign(2.f);

  sd::ops::reduce_dot_bp op;
  auto result = op.evaluate({&x, &y, &eps}, {}, {1});
  auto outputX = result.at(0);
  auto outputY = result.at(1);

  ASSERT_EQ(sd::Status::OK, result.status());
  ASSERT_TRUE(expX.equalsTo(outputX));
  ASSERT_TRUE(expY.equalsTo(outputY));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, cumsum_bp_1) {
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto eps = NDArrayFactory::create<double>('c', {3, 4});
  auto exp =
      NDArrayFactory::create<double>('c', {3, 4}, {12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f});
  x.linspace(1);
  eps.assign(1.f);

  sd::ops::cumsum_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {0, 0});
  auto output = result.at(0);

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, cumsum_bp_2) {
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto eps = NDArrayFactory::create<double>('c', {3, 4});
  auto exp =
      NDArrayFactory::create<double>('c', {3, 4}, {11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f});
  x.linspace(1);
  eps.assign(1.f);

  sd::ops::cumsum_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {1, 0});
  auto output = result.at(0);

  ASSERT_EQ(sd::Status::OK, result.status());

  ASSERT_TRUE(exp.isSameShape(output));
  ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests7, cumsum_bp_3) {
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto eps = NDArrayFactory::create<double>('c', {3, 4});
  auto exp = NDArrayFactory::create<double>('c', {3, 4});

  x.linspace(1);
  exp.linspace(0);
  eps.assign(1.f);

  sd::ops::cumsum_bp op;
  auto result = op.evaluate({&x, &eps}, {}, {1, 1});
  auto output = result.at(0);

  ASSERT_EQ(sd::Status::OK, result.status());
  ASSERT_TRUE(exp.equalsTo(output));
}