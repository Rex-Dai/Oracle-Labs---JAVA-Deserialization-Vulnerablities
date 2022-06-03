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


#ifndef FLATBUFFERS_GENERATED_UTILS_SD_GRAPH_H_
#define FLATBUFFERS_GENERATED_UTILS_SD_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

namespace sd {
namespace graph {

struct LongPair;
struct LongPairBuilder;

struct LongTriple;
struct LongTripleBuilder;

struct IntPair;
struct IntPairBuilder;

struct IntTriple;
struct IntTripleBuilder;

enum OpType {
  OpType_TRANSFORM_FLOAT = 0,
  OpType_TRANSFORM_SAME = 1,
  OpType_TRANSFORM_BOOL = 2,
  OpType_TRANSFORM_STRICT = 3,
  OpType_TRANSFORM_ANY = 4,
  OpType_REDUCE_FLOAT = 5,
  OpType_REDUCE_SAME = 6,
  OpType_REDUCE_LONG = 7,
  OpType_REDUCE_BOOL = 8,
  OpType_INDEX_REDUCE = 9,
  OpType_SCALAR = 10,
  OpType_SCALAR_BOOL = 11,
  OpType_BROADCAST = 12,
  OpType_BROADCAST_BOOL = 13,
  OpType_PAIRWISE = 14,
  OpType_PAIRWISE_BOOL = 15,
  OpType_REDUCE_3 = 16,
  OpType_SUMMARYSTATS = 17,
  OpType_SHAPE = 18,
  OpType_AGGREGATION = 19,
  OpType_RANDOM = 20,
  OpType_CUSTOM = 21,
  OpType_GRAPH = 22,
  OpType_VARIABLE = 40,
  OpType_BOOLEAN = 60,
  OpType_LOGIC = 119,
  OpType_MIN = OpType_TRANSFORM_FLOAT,
  OpType_MAX = OpType_LOGIC
};

inline const OpType (&EnumValuesOpType())[26] {
  static const OpType values[] = {
    OpType_TRANSFORM_FLOAT,
    OpType_TRANSFORM_SAME,
    OpType_TRANSFORM_BOOL,
    OpType_TRANSFORM_STRICT,
    OpType_TRANSFORM_ANY,
    OpType_REDUCE_FLOAT,
    OpType_REDUCE_SAME,
    OpType_REDUCE_LONG,
    OpType_REDUCE_BOOL,
    OpType_INDEX_REDUCE,
    OpType_SCALAR,
    OpType_SCALAR_BOOL,
    OpType_BROADCAST,
    OpType_BROADCAST_BOOL,
    OpType_PAIRWISE,
    OpType_PAIRWISE_BOOL,
    OpType_REDUCE_3,
    OpType_SUMMARYSTATS,
    OpType_SHAPE,
    OpType_AGGREGATION,
    OpType_RANDOM,
    OpType_CUSTOM,
    OpType_GRAPH,
    OpType_VARIABLE,
    OpType_BOOLEAN,
    OpType_LOGIC
  };
  return values;
}

inline const char * const *EnumNamesOpType() {
  static const char * const names[121] = {
    "TRANSFORM_FLOAT",
    "TRANSFORM_SAME",
    "TRANSFORM_BOOL",
    "TRANSFORM_STRICT",
    "TRANSFORM_ANY",
    "REDUCE_FLOAT",
    "REDUCE_SAME",
    "REDUCE_LONG",
    "REDUCE_BOOL",
    "INDEX_REDUCE",
    "SCALAR",
    "SCALAR_BOOL",
    "BROADCAST",
    "BROADCAST_BOOL",
    "PAIRWISE",
    "PAIRWISE_BOOL",
    "REDUCE_3",
    "SUMMARYSTATS",
    "SHAPE",
    "AGGREGATION",
    "RANDOM",
    "CUSTOM",
    "GRAPH",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "VARIABLE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "BOOLEAN",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "LOGIC",
    nullptr
  };
  return names;
}

inline const char *EnumNameOpType(OpType e) {
  if (flatbuffers::IsOutRange(e, OpType_TRANSFORM_FLOAT, OpType_LOGIC)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesOpType()[index];
}

enum InputType {
  InputType_UNDEFINED = 0,
  InputType_NUMERIC = 1,
  InputType_STRINGULAR = 2,
  InputType_NUMERIC_SET = 3,
  InputType_STRINGULAR_SET = 4,
  InputType_MIN = InputType_UNDEFINED,
  InputType_MAX = InputType_STRINGULAR_SET
};

inline const InputType (&EnumValuesInputType())[5] {
  static const InputType values[] = {
    InputType_UNDEFINED,
    InputType_NUMERIC,
    InputType_STRINGULAR,
    InputType_NUMERIC_SET,
    InputType_STRINGULAR_SET
  };
  return values;
}

inline const char * const *EnumNamesInputType() {
  static const char * const names[6] = {
    "UNDEFINED",
    "NUMERIC",
    "STRINGULAR",
    "NUMERIC_SET",
    "STRINGULAR_SET",
    nullptr
  };
  return names;
}

inline const char *EnumNameInputType(InputType e) {
  if (flatbuffers::IsOutRange(e, InputType_UNDEFINED, InputType_STRINGULAR_SET)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesInputType()[index];
}

enum OpClass {
  OpClass_TRANSFORM = 0,
  OpClass_REDUCTION = 1,
  OpClass_MULTIPLICATOR = 2,
  OpClass_GRAPH = 3,
  OpClass_CONDITIONAL = 4,
  OpClass_LOOP = 5,
  OpClass_MIN = OpClass_TRANSFORM,
  OpClass_MAX = OpClass_LOOP
};

inline const OpClass (&EnumValuesOpClass())[6] {
  static const OpClass values[] = {
    OpClass_TRANSFORM,
    OpClass_REDUCTION,
    OpClass_MULTIPLICATOR,
    OpClass_GRAPH,
    OpClass_CONDITIONAL,
    OpClass_LOOP
  };
  return values;
}

inline const char * const *EnumNamesOpClass() {
  static const char * const names[7] = {
    "TRANSFORM",
    "REDUCTION",
    "MULTIPLICATOR",
    "GRAPH",
    "CONDITIONAL",
    "LOOP",
    nullptr
  };
  return names;
}

inline const char *EnumNameOpClass(OpClass e) {
  if (flatbuffers::IsOutRange(e, OpClass_TRANSFORM, OpClass_LOOP)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesOpClass()[index];
}

struct LongPair FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef LongPairBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FIRST = 4,
    VT_SECOND = 6
  };
  int64_t first() const {
    return GetField<int64_t>(VT_FIRST, 0);
  }
  int64_t second() const {
    return GetField<int64_t>(VT_SECOND, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_FIRST) &&
           VerifyField<int64_t>(verifier, VT_SECOND) &&
           verifier.EndTable();
  }
};

struct LongPairBuilder {
  typedef LongPair Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int64_t first) {
    fbb_.AddElement<int64_t>(LongPair::VT_FIRST, first, 0);
  }
  void add_second(int64_t second) {
    fbb_.AddElement<int64_t>(LongPair::VT_SECOND, second, 0);
  }
  explicit LongPairBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  LongPairBuilder &operator=(const LongPairBuilder &);
  flatbuffers::Offset<LongPair> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<LongPair>(end);
    return o;
  }
};

inline flatbuffers::Offset<LongPair> CreateLongPair(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t first = 0,
    int64_t second = 0) {
  LongPairBuilder builder_(_fbb);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

struct LongTriple FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef LongTripleBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FIRST = 4,
    VT_SECOND = 6,
    VT_THIRD = 8
  };
  int64_t first() const {
    return GetField<int64_t>(VT_FIRST, 0);
  }
  int64_t second() const {
    return GetField<int64_t>(VT_SECOND, 0);
  }
  int64_t third() const {
    return GetField<int64_t>(VT_THIRD, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_FIRST) &&
           VerifyField<int64_t>(verifier, VT_SECOND) &&
           VerifyField<int64_t>(verifier, VT_THIRD) &&
           verifier.EndTable();
  }
};

struct LongTripleBuilder {
  typedef LongTriple Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int64_t first) {
    fbb_.AddElement<int64_t>(LongTriple::VT_FIRST, first, 0);
  }
  void add_second(int64_t second) {
    fbb_.AddElement<int64_t>(LongTriple::VT_SECOND, second, 0);
  }
  void add_third(int64_t third) {
    fbb_.AddElement<int64_t>(LongTriple::VT_THIRD, third, 0);
  }
  explicit LongTripleBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  LongTripleBuilder &operator=(const LongTripleBuilder &);
  flatbuffers::Offset<LongTriple> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<LongTriple>(end);
    return o;
  }
};

inline flatbuffers::Offset<LongTriple> CreateLongTriple(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t first = 0,
    int64_t second = 0,
    int64_t third = 0) {
  LongTripleBuilder builder_(_fbb);
  builder_.add_third(third);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

struct IntPair FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef IntPairBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FIRST = 4,
    VT_SECOND = 6
  };
  int32_t first() const {
    return GetField<int32_t>(VT_FIRST, 0);
  }
  int32_t second() const {
    return GetField<int32_t>(VT_SECOND, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_FIRST) &&
           VerifyField<int32_t>(verifier, VT_SECOND) &&
           verifier.EndTable();
  }
};

struct IntPairBuilder {
  typedef IntPair Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int32_t first) {
    fbb_.AddElement<int32_t>(IntPair::VT_FIRST, first, 0);
  }
  void add_second(int32_t second) {
    fbb_.AddElement<int32_t>(IntPair::VT_SECOND, second, 0);
  }
  explicit IntPairBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  IntPairBuilder &operator=(const IntPairBuilder &);
  flatbuffers::Offset<IntPair> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<IntPair>(end);
    return o;
  }
};

inline flatbuffers::Offset<IntPair> CreateIntPair(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t first = 0,
    int32_t second = 0) {
  IntPairBuilder builder_(_fbb);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

struct IntTriple FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef IntTripleBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FIRST = 4,
    VT_SECOND = 6,
    VT_THIRD = 8
  };
  int32_t first() const {
    return GetField<int32_t>(VT_FIRST, 0);
  }
  int32_t second() const {
    return GetField<int32_t>(VT_SECOND, 0);
  }
  int32_t third() const {
    return GetField<int32_t>(VT_THIRD, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_FIRST) &&
           VerifyField<int32_t>(verifier, VT_SECOND) &&
           VerifyField<int32_t>(verifier, VT_THIRD) &&
           verifier.EndTable();
  }
};

struct IntTripleBuilder {
  typedef IntTriple Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_first(int32_t first) {
    fbb_.AddElement<int32_t>(IntTriple::VT_FIRST, first, 0);
  }
  void add_second(int32_t second) {
    fbb_.AddElement<int32_t>(IntTriple::VT_SECOND, second, 0);
  }
  void add_third(int32_t third) {
    fbb_.AddElement<int32_t>(IntTriple::VT_THIRD, third, 0);
  }
  explicit IntTripleBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  IntTripleBuilder &operator=(const IntTripleBuilder &);
  flatbuffers::Offset<IntTriple> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<IntTriple>(end);
    return o;
  }
};

inline flatbuffers::Offset<IntTriple> CreateIntTriple(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t first = 0,
    int32_t second = 0,
    int32_t third = 0) {
  IntTripleBuilder builder_(_fbb);
  builder_.add_third(third);
  builder_.add_second(second);
  builder_.add_first(first);
  return builder_.Finish();
}

}  // namespace graph
}  // namespace sd

#endif  // FLATBUFFERS_GENERATED_UTILS_SD_GRAPH_H_
