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
// @author Yurii Shyrma, created on 06.03.2018
//

#ifndef LIBND4J_CONVO_OPS_H
#define LIBND4J_CONVO_OPS_H

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_conv2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/op_boilerplate.h>

#include <memory>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(conv2d, 2, 1, false, 0, 9) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_NULLIFIED(0);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  int sH = INT_ARG(2);                                               // strides height
  int sW = INT_ARG(3);                                               // strides width
  int pH = INT_ARG(4);                                               // paddings height
  int pW = INT_ARG(5);                                               // paddings width
  int dH = INT_ARG(6);                                               // dilations height
  int dW = INT_ARG(7);                                               // dilations width
  int isSameMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));  // filter(kernel) height
  int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));  // filter(kernel) width

  int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(
        bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
        "CUSTOM CONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !",
        oC, bias->rankOf(), bias->lengthOf());

  ConvolutionUtils::conv2d(block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW,
                           wFormat);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(conv2d) {
  auto inputShapeInfo = inputShape->at(0);    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weightsShapeInfo = inputShape->at(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto biasShapeInfo = block.width() > 2 ? inputShape->at(2) : nullptr;  // [oC]

  // output [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  int sH = INT_ARG(2);                                               // strides height
  int sW = INT_ARG(3);                                               // strides width
  int pH = INT_ARG(4);                                               // paddings height
  int pW = INT_ARG(5);                                               // paddings width
  int dH = INT_ARG(6);                                               // dilations height
  int dW = INT_ARG(7);                                               // dilations width
  int isSameMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0));  // filter(kernel) height
  int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 1));  // filter(kernel) width

  const int rank = 4;  // 4

  REQUIRE_TRUE(inputShapeInfo[0] == rank, 0,
               "CUSTOM CONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank,
               inputShapeInfo[0]);
  REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0,
               "CUSTOM CONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank,
               weightsShapeInfo[0]);

  int indIOioC, indIiH, indWoC(0 == wFormat ? 3 : 0);
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
  }

  const int bS = inputShapeInfo[1];             // batch size
  const int iH = inputShapeInfo[indIiH + 1];    // input height
  const int iW = inputShapeInfo[indIiH + 2];    // input width
  const int iC = inputShapeInfo[indIOioC + 1];  // input channels
  const int oC = weightsShapeInfo[indWoC + 1];  // output channels

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(weightsShapeInfo, expectedWeightsShape), 0,
               "CUSTOM CONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
  if (biasShapeInfo)
    REQUIRE_TRUE(
        biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0,
        "CUSTOM CONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !",
        oC, biasShapeInfo[0], shape::length(biasShapeInfo));

  sd::LongType* outputShapeInfo = nullptr;
  ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);

  int oH, oW;  // output height, width
  ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  outputShapeInfo[0] = rank;
  outputShapeInfo[1] = bS;

  if (isNCHW) {
    outputShapeInfo[2] = oC;
    outputShapeInfo[3] = oH;
    outputShapeInfo[4] = oW;
  } else {
    outputShapeInfo[2] = oH;
    outputShapeInfo[3] = oW;
    outputShapeInfo[4] = oC;
  }

  ShapeUtils::updateStridesAndType(outputShapeInfo, weightsShapeInfo, shape::order(inputShapeInfo));

  return SHAPELIST(CONSTANT(outputShapeInfo));
}

DECLARE_TYPES(conv2d) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_TYPES(conv2d_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(conv2d_bp, 3, 2, false, 0, 9) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  auto gradW = OUTPUT_NULLIFIED(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_NULLIFIED(2) : nullptr;  // [oC]

  int kH = INT_ARG(0);                                               // filter(kernel) height
  int kW = INT_ARG(1);                                               // filter(kernel) width
  int sH = INT_ARG(2);                                               // strides height
  int sW = INT_ARG(3);                                               // strides width
  int pH = INT_ARG(4);                                               // paddings height
  int pW = INT_ARG(5);                                               // paddings width
  int dH = INT_ARG(6);                                               // dilations height
  int dW = INT_ARG(7);                                               // dilations width
  int isSameMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "CUSTOM CONV2D_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "CUSTOM CONV2D_BP OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(
      gradO->rankOf() == 4, 0,
      "CUSTOM CONV2D_BP OP: rank of output's gradients (next epsilon) array must be equal to 4, but got %i instead !",
      gradO->rankOf());

  int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  int trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(
      gradO->isSameShape(expectedGradOShape), 0,
      "CUSTOM CONV2D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM CONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  ConvolutionUtils::conv2dBP(block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW,
                             isSameMode, isNCHW, wFormat);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(conv2d_bp) {
  auto inputShapeInfo = inputShape->at(0);    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weightsShapeInfo = inputShape->at(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto biasShapeInfo = block.width() > 3 ? inputShape->at(2) : nullptr;  // [oC]
  auto gradOShapeInfo = block.width() > 3
                            ? inputShape->at(3)
                            : inputShape->at(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  const int rank = 4;

  REQUIRE_TRUE(inputShapeInfo[0] == rank, 0,
               "CUSTOM CONV2D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank,
               inputShapeInfo[0]);
  REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0,
               "CUSTOM CONV2D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank,
               weightsShapeInfo[0]);
  REQUIRE_TRUE(
      gradOShapeInfo[0] == rank, 0,
      "CUSTOM CONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !",
      rank, gradOShapeInfo[0]);

  const int kH = INT_ARG(0);                                               // filter(kernel) height
  const int kW = INT_ARG(1);                                               // filter(kernel) width
  const int sH = INT_ARG(2);                                               // strides height
  const int sW = INT_ARG(3);                                               // strides width
  const int pH = INT_ARG(4);                                               // paddings height
  const int pW = INT_ARG(5);                                               // paddings width
  const int dH = INT_ARG(6);                                               // dilations height
  const int dW = INT_ARG(7);                                               // dilations width
  const int isSameMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  const int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  const int wFormat = block.getIArguments()->size() > 10
                          ? INT_ARG(10)
                          : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  int indIOioC, indIiH, indOoH, indWoC(0 == wFormat ? 3 : 0);
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
    indOoH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
    indOoH = 2;
  }

  const int bS = inputShapeInfo[1];             // batch size
  const int iH = inputShapeInfo[indIiH + 1];    // input height
  const int iW = inputShapeInfo[indIiH + 2];    // input width
  const int iC = inputShapeInfo[indIOioC + 1];  // input channels
  const int oC = weightsShapeInfo[indWoC + 1];  // output channels

  int trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(
      ShapeUtils::areShapesEqual(gradOShapeInfo, expectedGradOShape), 0,
      "CUSTOM CONV2D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(weightsShapeInfo, expectedWeightsShape), 0,
               "CUSTOM CONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
  if (biasShapeInfo)
    REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0,
                 "CUSTOM CONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, biasShapeInfo[0], shape::length(biasShapeInfo));

  auto gradIshapeInfo =
      ShapeBuilders::copyShapeInfoAndType(inputShapeInfo, gradOShapeInfo, false, block.getWorkspace());
  auto gradWshapeInfo =
      ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, gradOShapeInfo, false, block.getWorkspace());

  if (biasShapeInfo) {
    auto gradBshapeInfo =
        ShapeBuilders::copyShapeInfoAndType(biasShapeInfo, gradOShapeInfo, false, block.getWorkspace());
    return SHAPELIST(CONSTANT(gradIshapeInfo), CONSTANT(gradWshapeInfo), CONSTANT(gradBshapeInfo));
  }

  return SHAPELIST(CONSTANT(gradIshapeInfo), CONSTANT(gradWshapeInfo));
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(conv2d_input_bp, 3, 1, false, 0, 9) {
  auto gradIShape = INPUT_VARIABLE(0);  // [4]
  auto weights = INPUT_VARIABLE(1);     // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradO = INPUT_VARIABLE(2);       // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  int kH = INT_ARG(0);                                               // filter(kernel) height
  int kW = INT_ARG(1);                                               // filter(kernel) width
  int sH = INT_ARG(2);                                               // strides height
  int sW = INT_ARG(3);                                               // strides width
  int pH = INT_ARG(4);                                               // paddings height
  int pW = INT_ARG(5);                                               // paddings width
  int dH = INT_ARG(6);                                               // dilations height
  int dW = INT_ARG(7);                                               // dilations width
  int isSameMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  const int rank = gradO->rankOf();

  REQUIRE_TRUE(weights->rankOf() == rank, 0,
               "CUSTOM CONV2D_INPUT_BP OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradIShape->rankOf() == 1, 0,
               "CUSTOM CONV2D_INPUT_BP OP: rank of array with output shape must be equal to 1, but got %i instead !",
               gradIShape->rankOf());
  REQUIRE_TRUE(gradIShape->lengthOf() == rank, 0,
               "CUSTOM CONV2D_INPUT_BP OP: length of array with output shape must be equal to 4, but got %i instead !",
               gradIShape->lengthOf());

  // create empty conv2d input array
  std::vector<sd::LongType> gradIShapeAsVector(rank);
  for (int i = 0; i < rank; ++i) gradIShapeAsVector[i] = gradIShape->e<sd::LongType>(i);
  NDArray input(gradO->ordering(), gradIShapeAsVector, gradO->dataType(), block.launchContext());

  int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  int trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM CONV2D_INPUT_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but "
               "got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV2D_INPUT_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

  ConvolutionUtils::conv2dBP(block, &input, weights, nullptr, gradO, gradI, nullptr, nullptr, kH, kW, sH, sW, pH, pW,
                             dH, dW, isSameMode, isNCHW, wFormat);

  return sd::Status::OK;
}

DECLARE_TYPES(conv2d_input_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(conv2d_input_bp) {
  auto gradIShapeShapeInfo = inputShape->at(0);  // [4]
  auto weightsShapeInfo = inputShape->at(1);     // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradOShapeInfo = inputShape->at(2);       // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  const int rank = 4;

  REQUIRE_TRUE(gradIShapeShapeInfo[0] == 1, 0,
               "CUSTOM CONV2D_INPUT_BP OP: rank of array with output shape must be equal to %i, but got %i instead !",
               1, gradIShapeShapeInfo[0]);
  REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0,
               "CUSTOM CONV2D_INPUT_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank,
               weightsShapeInfo[0]);
  REQUIRE_TRUE(gradOShapeInfo[0] == rank, 0,
               "CUSTOM CONV2D_INPUT_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got "
               "%i instead !",
               rank, gradOShapeInfo[0]);

  const int kH = INT_ARG(0);                                               // filter(kernel) height
  const int kW = INT_ARG(1);                                               // filter(kernel) width
  const int sH = INT_ARG(2);                                               // strides height
  const int sW = INT_ARG(3);                                               // strides width
  const int pH = INT_ARG(4);                                               // paddings height
  const int pW = INT_ARG(5);                                               // paddings width
  const int dH = INT_ARG(6);                                               // dilations height
  const int dW = INT_ARG(7);                                               // dilations width
  const int isSameMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  const int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  const int wFormat = block.getIArguments()->size() > 10
                          ? INT_ARG(10)
                          : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  int indIOioC, indIiH, indWoC(0 == wFormat ? 3 : 0), indOoH;
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
    indOoH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
    indOoH = 2;
  }

  std::vector<sd::LongType> gradIShape = INPUT_VARIABLE(0)->template asVectorT<sd::LongType>();

  const int bS = gradIShape[0];                 // batch size
  const int iH = gradIShape[indIiH];            // input height
  const int iW = gradIShape[indIiH + 1];        // input width
  const int iC = gradIShape[indIOioC];          // input channels
  const int oC = weightsShapeInfo[indWoC + 1];  // output channels

  int trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(gradOShapeInfo, expectedGradOShape), 0,
               "CUSTOM CONV2D_INPUT_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but "
               "got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(),
               ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
  REQUIRE_TRUE(ShapeUtils::areShapesEqual(weightsShapeInfo, expectedWeightsShape), 0,
               "CUSTOM CONV2D_INPUT_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());

  sd::LongType* gradIshapeInfo(nullptr);
  ALLOCATE(gradIshapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);

  gradIshapeInfo[0] = rank;
  gradIshapeInfo[1] = bS;

  if (isNCHW) {
    gradIshapeInfo[2] = iC;
    gradIshapeInfo[3] = iH;
    gradIshapeInfo[4] = iW;
  } else {
    gradIshapeInfo[2] = iH;
    gradIshapeInfo[3] = iW;
    gradIshapeInfo[4] = iC;
  }

  ShapeUtils::updateStridesAndType(gradIshapeInfo, gradOShapeInfo, shape::order(gradOShapeInfo));

  return SHAPELIST(CONSTANT(gradIshapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif

#endif  // LIBND4J_CONVO_OPS_H