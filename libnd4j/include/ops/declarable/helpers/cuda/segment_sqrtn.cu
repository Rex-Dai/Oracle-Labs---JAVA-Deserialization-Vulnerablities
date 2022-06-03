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
//  @author GS <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>

namespace sd {
namespace ops {
namespace helpers {
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentSqrtNLinearKernel(T* input, sd::LongType const* inputShape, I* indices,
                                                       sd::LongType const* indicesShape, int* starts, int* lengths,
                                                       sd::LongType numOfClasses, T* output,
                                                       sd::LongType const* outputShape) {
  __shared__ sd::LongType xLen, zLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);
  }
  __syncthreads();

  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (auto idx = start; idx < xLen; idx += step) {
    auto yIndex = shape::getIndexOffset(idx, indicesShape);
    auto segment = indices[yIndex];
    auto zIndex = shape::getIndexOffset(segment, outputShape);
    if (lengths[segment] == 0) continue;
    auto xIndex = shape::getIndexOffset(idx, inputShape);

    sd::math::atomics::sd_atomicAdd(&output[zIndex], input[xIndex] / sd::math::sd_sqrt<int, T>(lengths[segment]));
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentSqrtN kernel
template <typename T, typename I>
static SD_KERNEL void segmentSqrtNTadKernel(T* inputBuf, sd::LongType const* inputShape, sd::LongType const* inputTads,
                                            sd::LongType const* inputTadOffsets, I* indices, int* starts, int* lengths,
                                            sd::LongType numOfClasses, void* outputBuf, sd::LongType const* outputShape,
                                            sd::LongType const* outputTads, sd::LongType const* outputTadOffsets) {
  __shared__ sd::LongType len, total;

  if (threadIdx.x == 0) {
    total = shape::sizeAt(inputShape, 0);
    len = shape::length(inputTads);
  }
  __syncthreads();

  for (auto idx = blockIdx.x; idx < total; idx += gridDim.x) {
    auto segment = indices[idx];
    auto x = inputBuf + inputTadOffsets[idx];
    auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    auto start = starts[segment];
    auto finish = start + lengths[segment];

    for (auto e = threadIdx.x; e < len; e += blockDim.x) {
      auto xIndex = shape::getIndexOffset(e, inputTads);
      auto zIndex = shape::getIndexOffset(e, outputTads);
      sd::math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex] / sd::math::sd_sqrt<int, T>(lengths[segment]));
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentSqrtNFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                         sd::LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  //        NDArray classes = NDArrayFactory::create<int>('c', {numOfClasses, 2});
  NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numOfClasses}, context);
  //        NDArray row = NDArrayFactory::create<int>('c', {1, 2}, {(int)indices->lengthOf(), (int)0});
  //        classes.applyTrueBroadcast(sd::BroadcastOpsTuple::Assign(), &row, &classes);
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);
  //        dim3 dims(numOfClasses, indices->lengthOf(), numOfClasses * 32 + 32);
  dim3 dims(128, 256, 256);
  //        int* classesBuf = reinterpret_cast<int*>(classes.specialBuffer());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
  int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());
  output->nullify();
  if (input->isVector()) {
    unsortedSegmentSqrtNLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), indices->dataBuffer()->specialAsT<I>(),
        indices->specialShapeInfo(), begins, lengths, numOfClasses, output->dataBuffer()->specialAsT<T>(),
        output->specialShapeInfo());
  } else {
    output->nullify();
    std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX.specialShapeInfo();
    auto inputTadOffsets = packX.specialOffsets();
    auto outputTads = packZ.specialShapeInfo();
    auto outputTadOffsets = packZ.specialOffsets();
    dims.x = input->sizeAt(0);
    segmentSqrtNTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        indices->dataBuffer()->specialAsT<I>(), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets);
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentSqrtNFunctor(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                 sd::LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSqrtNFunctor_,
                        (context, input, indices, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentSqrtNBPLinearKernel(void* inputBuf, sd::LongType const* inputShape, void* eps,
                                                 sd::LongType const* epsShape, void* indicesBuf,
                                                 sd::LongType const* indicesShape, int* lengths, void* outputBuf,
                                                 sd::LongType const* outputShape) {
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ sd::LongType xLen, gradLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradOut = reinterpret_cast<T*>(eps);
    gradLen = shape::length(epsShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  for (auto e = start; e < xLen; e += step) {
    auto zOffset = shape::getIndexOffset(e, outputShape);
    auto xOffset = shape::getIndexOffset(e, inputShape);
    auto yOffset = shape::getIndexOffset(e, indicesShape);
    auto classIndex = y[yOffset];
    auto gradOffsetO = shape::getIndexOffset(classIndex, epsShape);

    z[zOffset] = T(gradOut[gradOffsetO] / math::sd_sqrt<int, float>(lengths[classIndex]));
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void segmentSqrtNBPTadKernel(void* inputBuf, sd::LongType const* inputShape, void* eps,
                                              sd::LongType const* epsShape, void* indicesBuf,
                                              sd::LongType const* indicesShape, int* lengths, void* outputBuf,
                                              sd::LongType const* outputShape, sd::LongType const* inputTad,
                                              sd::LongType const* inputOffsets, sd::LongType const* gradOutTad,
                                              sd::LongType const* gradOutOffsets, sd::LongType const* outTad,
                                              sd::LongType const* outOffsets) {
  __shared__ T* x;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ sd::LongType xLen, yLen, gradLen, currentLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    yLen = shape::length(indicesShape);
    gradOut = reinterpret_cast<T*>(eps);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    //            auto yIndex = shape::getIndexOffset(i, indicesShape);
    auto segment = y[i];  // yIndex];
    T* currentOut = z + outOffsets[i];
    T* outGrad = gradOut + gradOutOffsets[segment];

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      auto zIndex = shape::getIndexOffset(e, outTad);
      auto gradIndex = shape::getIndexOffset(e, gradOutTad);
      if (lengths[segment] > 0)
        currentOut[zIndex] = T(outGrad[gradIndex] / math::sd_sqrt<int, float>(lengths[segment]));
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static sd::Status unsortedSegmentSqrtNFunctorBP_(sd::LaunchContext* context, NDArray* input, NDArray* indices,
                                                 NDArray* gradOut, sd::LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  auto numClasses = indices->e<int>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<int>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<int>('c', {numClasses}, context);

  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);
  dim3 dims(numClasses, indices->lengthOf(), numClasses * 32 + 32);
  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  int* begins = reinterpret_cast<int*>(classesRangesBegs.specialBuffer());
  int* lengths = reinterpret_cast<int*>(classesRangesLens.specialBuffer());

  if (input->isVector()) {
    sd::LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
    segmentSqrtNBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo());
  } else {
    std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    //            auto packGradIn = sd::ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(),
    //            dimensions);
    auto packGradOut = sd::ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    auto inputTads = packX.specialShapeInfo();
    auto inputTadOffsets = packX.specialOffsets();
    auto outputTads = packZ.specialShapeInfo();
    auto outputTadOffsets = packZ.specialOffsets();
    auto gradOutTads = packGradOut.specialShapeInfo();
    auto gradOutTadOffsets = packGradOut.specialOffsets();

    segmentSqrtNBPTadKernel<T, I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
        outputTadOffsets);
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});

  return sd::Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
sd::Status unsortedSegmentSqrtNFunctorBP(sd::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                         sd::LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSqrtNFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd