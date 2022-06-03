/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


#ifndef FLATBUFFERS_GENERATED_GRAPH_SD_GRAPH_H_
#define FLATBUFFERS_GENERATED_GRAPH_SD_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

#include "array_generated.h"
#include "config_generated.h"
#include "node_generated.h"
#include "properties_generated.h"
#include "request_generated.h"
#include "result_generated.h"
#include "utils_generated.h"
#include "variable_generated.h"

namespace sd {
namespace graph {

struct UpdaterState;
struct UpdaterStateBuilder;

struct FlatGraph;
struct FlatGraphBuilder;

struct FlatDropRequest;
struct FlatDropRequestBuilder;

struct FlatResponse;
struct FlatResponseBuilder;

struct UpdaterState FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef UpdaterStateBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_PARAMNAME = 4,
    VT_UPDATERSTATEKEYS = 6,
    VT_UPDATERSTATEVALUES = 8
  };
  const flatbuffers::String *paramName() const {
    return GetPointer<const flatbuffers::String *>(VT_PARAMNAME);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *updaterStateKeys() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_UPDATERSTATEKEYS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatArray>> *updaterStateValues() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatArray>> *>(VT_UPDATERSTATEVALUES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_PARAMNAME) &&
           verifier.VerifyString(paramName()) &&
           VerifyOffset(verifier, VT_UPDATERSTATEKEYS) &&
           verifier.VerifyVector(updaterStateKeys()) &&
           verifier.VerifyVectorOfStrings(updaterStateKeys()) &&
           VerifyOffset(verifier, VT_UPDATERSTATEVALUES) &&
           verifier.VerifyVector(updaterStateValues()) &&
           verifier.VerifyVectorOfTables(updaterStateValues()) &&
           verifier.EndTable();
  }
};

struct UpdaterStateBuilder {
  typedef UpdaterState Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_paramName(flatbuffers::Offset<flatbuffers::String> paramName) {
    fbb_.AddOffset(UpdaterState::VT_PARAMNAME, paramName);
  }
  void add_updaterStateKeys(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> updaterStateKeys) {
    fbb_.AddOffset(UpdaterState::VT_UPDATERSTATEKEYS, updaterStateKeys);
  }
  void add_updaterStateValues(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatArray>>> updaterStateValues) {
    fbb_.AddOffset(UpdaterState::VT_UPDATERSTATEVALUES, updaterStateValues);
  }
  explicit UpdaterStateBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  UpdaterStateBuilder &operator=(const UpdaterStateBuilder &);
  flatbuffers::Offset<UpdaterState> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<UpdaterState>(end);
    return o;
  }
};

inline flatbuffers::Offset<UpdaterState> CreateUpdaterState(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> paramName = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> updaterStateKeys = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatArray>>> updaterStateValues = 0) {
  UpdaterStateBuilder builder_(_fbb);
  builder_.add_updaterStateValues(updaterStateValues);
  builder_.add_updaterStateKeys(updaterStateKeys);
  builder_.add_paramName(paramName);
  return builder_.Finish();
}

inline flatbuffers::Offset<UpdaterState> CreateUpdaterStateDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *paramName = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *updaterStateKeys = nullptr,
    const std::vector<flatbuffers::Offset<sd::graph::FlatArray>> *updaterStateValues = nullptr) {
  auto paramName__ = paramName ? _fbb.CreateString(paramName) : 0;
  auto updaterStateKeys__ = updaterStateKeys ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*updaterStateKeys) : 0;
  auto updaterStateValues__ = updaterStateValues ? _fbb.CreateVector<flatbuffers::Offset<sd::graph::FlatArray>>(*updaterStateValues) : 0;
  return sd::graph::CreateUpdaterState(
      _fbb,
      paramName__,
      updaterStateKeys__,
      updaterStateValues__);
}

struct FlatGraph FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FlatGraphBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ID = 4,
    VT_VARIABLES = 6,
    VT_NODES = 8,
    VT_OUTPUTS = 10,
    VT_CONFIGURATION = 12,
    VT_PLACEHOLDERS = 14,
    VT_LOSSVARIABLES = 16,
    VT_TRAININGCONFIG = 18,
    VT_UPDATERSTATE = 20
  };
  int64_t id() const {
    return GetField<int64_t>(VT_ID, 0);
  }
  const flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatVariable>> *variables() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatVariable>> *>(VT_VARIABLES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatNode>> *nodes() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatNode>> *>(VT_NODES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<sd::graph::IntPair>> *outputs() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<sd::graph::IntPair>> *>(VT_OUTPUTS);
  }
  const sd::graph::FlatConfiguration *configuration() const {
    return GetPointer<const sd::graph::FlatConfiguration *>(VT_CONFIGURATION);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *placeholders() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_PLACEHOLDERS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *lossVariables() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_LOSSVARIABLES);
  }
  const flatbuffers::String *trainingConfig() const {
    return GetPointer<const flatbuffers::String *>(VT_TRAININGCONFIG);
  }
  const flatbuffers::Vector<flatbuffers::Offset<sd::graph::UpdaterState>> *updaterState() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<sd::graph::UpdaterState>> *>(VT_UPDATERSTATE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_ID) &&
           VerifyOffset(verifier, VT_VARIABLES) &&
           verifier.VerifyVector(variables()) &&
           verifier.VerifyVectorOfTables(variables()) &&
           VerifyOffset(verifier, VT_NODES) &&
           verifier.VerifyVector(nodes()) &&
           verifier.VerifyVectorOfTables(nodes()) &&
           VerifyOffset(verifier, VT_OUTPUTS) &&
           verifier.VerifyVector(outputs()) &&
           verifier.VerifyVectorOfTables(outputs()) &&
           VerifyOffset(verifier, VT_CONFIGURATION) &&
           verifier.VerifyTable(configuration()) &&
           VerifyOffset(verifier, VT_PLACEHOLDERS) &&
           verifier.VerifyVector(placeholders()) &&
           verifier.VerifyVectorOfStrings(placeholders()) &&
           VerifyOffset(verifier, VT_LOSSVARIABLES) &&
           verifier.VerifyVector(lossVariables()) &&
           verifier.VerifyVectorOfStrings(lossVariables()) &&
           VerifyOffset(verifier, VT_TRAININGCONFIG) &&
           verifier.VerifyString(trainingConfig()) &&
           VerifyOffset(verifier, VT_UPDATERSTATE) &&
           verifier.VerifyVector(updaterState()) &&
           verifier.VerifyVectorOfTables(updaterState()) &&
           verifier.EndTable();
  }
};

struct FlatGraphBuilder {
  typedef FlatGraph Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(int64_t id) {
    fbb_.AddElement<int64_t>(FlatGraph::VT_ID, id, 0);
  }
  void add_variables(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatVariable>>> variables) {
    fbb_.AddOffset(FlatGraph::VT_VARIABLES, variables);
  }
  void add_nodes(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatNode>>> nodes) {
    fbb_.AddOffset(FlatGraph::VT_NODES, nodes);
  }
  void add_outputs(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::IntPair>>> outputs) {
    fbb_.AddOffset(FlatGraph::VT_OUTPUTS, outputs);
  }
  void add_configuration(flatbuffers::Offset<sd::graph::FlatConfiguration> configuration) {
    fbb_.AddOffset(FlatGraph::VT_CONFIGURATION, configuration);
  }
  void add_placeholders(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> placeholders) {
    fbb_.AddOffset(FlatGraph::VT_PLACEHOLDERS, placeholders);
  }
  void add_lossVariables(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> lossVariables) {
    fbb_.AddOffset(FlatGraph::VT_LOSSVARIABLES, lossVariables);
  }
  void add_trainingConfig(flatbuffers::Offset<flatbuffers::String> trainingConfig) {
    fbb_.AddOffset(FlatGraph::VT_TRAININGCONFIG, trainingConfig);
  }
  void add_updaterState(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::UpdaterState>>> updaterState) {
    fbb_.AddOffset(FlatGraph::VT_UPDATERSTATE, updaterState);
  }
  explicit FlatGraphBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatGraphBuilder &operator=(const FlatGraphBuilder &);
  flatbuffers::Offset<FlatGraph> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatGraph>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatGraph> CreateFlatGraph(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t id = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatVariable>>> variables = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::FlatNode>>> nodes = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::IntPair>>> outputs = 0,
    flatbuffers::Offset<sd::graph::FlatConfiguration> configuration = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> placeholders = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> lossVariables = 0,
    flatbuffers::Offset<flatbuffers::String> trainingConfig = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<sd::graph::UpdaterState>>> updaterState = 0) {
  FlatGraphBuilder builder_(_fbb);
  builder_.add_id(id);
  builder_.add_updaterState(updaterState);
  builder_.add_trainingConfig(trainingConfig);
  builder_.add_lossVariables(lossVariables);
  builder_.add_placeholders(placeholders);
  builder_.add_configuration(configuration);
  builder_.add_outputs(outputs);
  builder_.add_nodes(nodes);
  builder_.add_variables(variables);
  return builder_.Finish();
}

inline flatbuffers::Offset<FlatGraph> CreateFlatGraphDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t id = 0,
    const std::vector<flatbuffers::Offset<sd::graph::FlatVariable>> *variables = nullptr,
    const std::vector<flatbuffers::Offset<sd::graph::FlatNode>> *nodes = nullptr,
    const std::vector<flatbuffers::Offset<sd::graph::IntPair>> *outputs = nullptr,
    flatbuffers::Offset<sd::graph::FlatConfiguration> configuration = 0,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *placeholders = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *lossVariables = nullptr,
    const char *trainingConfig = nullptr,
    const std::vector<flatbuffers::Offset<sd::graph::UpdaterState>> *updaterState = nullptr) {
  auto variables__ = variables ? _fbb.CreateVector<flatbuffers::Offset<sd::graph::FlatVariable>>(*variables) : 0;
  auto nodes__ = nodes ? _fbb.CreateVector<flatbuffers::Offset<sd::graph::FlatNode>>(*nodes) : 0;
  auto outputs__ = outputs ? _fbb.CreateVector<flatbuffers::Offset<sd::graph::IntPair>>(*outputs) : 0;
  auto placeholders__ = placeholders ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*placeholders) : 0;
  auto lossVariables__ = lossVariables ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*lossVariables) : 0;
  auto trainingConfig__ = trainingConfig ? _fbb.CreateString(trainingConfig) : 0;
  auto updaterState__ = updaterState ? _fbb.CreateVector<flatbuffers::Offset<sd::graph::UpdaterState>>(*updaterState) : 0;
  return sd::graph::CreateFlatGraph(
      _fbb,
      id,
      variables__,
      nodes__,
      outputs__,
      configuration,
      placeholders__,
      lossVariables__,
      trainingConfig__,
      updaterState__);
}

struct FlatDropRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FlatDropRequestBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ID = 4
  };
  int64_t id() const {
    return GetField<int64_t>(VT_ID, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_ID) &&
           verifier.EndTable();
  }
};

struct FlatDropRequestBuilder {
  typedef FlatDropRequest Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(int64_t id) {
    fbb_.AddElement<int64_t>(FlatDropRequest::VT_ID, id, 0);
  }
  explicit FlatDropRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatDropRequestBuilder &operator=(const FlatDropRequestBuilder &);
  flatbuffers::Offset<FlatDropRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatDropRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatDropRequest> CreateFlatDropRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t id = 0) {
  FlatDropRequestBuilder builder_(_fbb);
  builder_.add_id(id);
  return builder_.Finish();
}

struct FlatResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FlatResponseBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_STATUS = 4
  };
  int32_t status() const {
    return GetField<int32_t>(VT_STATUS, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_STATUS) &&
           verifier.EndTable();
  }
};

struct FlatResponseBuilder {
  typedef FlatResponse Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_status(int32_t status) {
    fbb_.AddElement<int32_t>(FlatResponse::VT_STATUS, status, 0);
  }
  explicit FlatResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatResponseBuilder &operator=(const FlatResponseBuilder &);
  flatbuffers::Offset<FlatResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatResponse> CreateFlatResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t status = 0) {
  FlatResponseBuilder builder_(_fbb);
  builder_.add_status(status);
  return builder_.Finish();
}

inline const sd::graph::FlatGraph *GetFlatGraph(const void *buf) {
  return flatbuffers::GetRoot<sd::graph::FlatGraph>(buf);
}

inline const sd::graph::FlatGraph *GetSizePrefixedFlatGraph(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<sd::graph::FlatGraph>(buf);
}

inline bool VerifyFlatGraphBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<sd::graph::FlatGraph>(nullptr);
}

inline bool VerifySizePrefixedFlatGraphBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<sd::graph::FlatGraph>(nullptr);
}

inline void FinishFlatGraphBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<sd::graph::FlatGraph> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedFlatGraphBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<sd::graph::FlatGraph> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace graph
}  // namespace sd

#endif  // FLATBUFFERS_GENERATED_GRAPH_SD_GRAPH_H_
