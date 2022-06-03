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
package org.nd4j.graph;

import java.nio.*;
import java.lang.*;
import java.nio.ByteOrder;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class UIVariable extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static UIVariable getRootAsUIVariable(ByteBuffer _bb) { return getRootAsUIVariable(_bb, new UIVariable()); }
  public static UIVariable getRootAsUIVariable(ByteBuffer _bb, UIVariable obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public UIVariable __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public org.nd4j.graph.IntPair id() { return id(new org.nd4j.graph.IntPair()); }
  public org.nd4j.graph.IntPair id(org.nd4j.graph.IntPair obj) { int o = __offset(4); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }
  public String name() { int o = __offset(6); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer nameAsByteBuffer() { return __vector_as_bytebuffer(6, 1); }
  public ByteBuffer nameInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 6, 1); }
  public byte type() { int o = __offset(8); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public byte datatype() { int o = __offset(10); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public long shape(int j) { int o = __offset(12); return o != 0 ? bb.getLong(__vector(o) + j * 8) : 0; }
  public int shapeLength() { int o = __offset(12); return o != 0 ? __vector_len(o) : 0; }
  public LongVector shapeVector() { return shapeVector(new LongVector()); }
  public LongVector shapeVector(LongVector obj) { int o = __offset(12); return o != 0 ? obj.__assign(__vector(o), bb) : null; }
  public ByteBuffer shapeAsByteBuffer() { return __vector_as_bytebuffer(12, 8); }
  public ByteBuffer shapeInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 12, 8); }
  public String controlDeps(int j) { int o = __offset(14); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int controlDepsLength() { int o = __offset(14); return o != 0 ? __vector_len(o) : 0; }
  public StringVector controlDepsVector() { return controlDepsVector(new StringVector()); }
  public StringVector controlDepsVector(StringVector obj) { int o = __offset(14); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String outputOfOp() { int o = __offset(16); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer outputOfOpAsByteBuffer() { return __vector_as_bytebuffer(16, 1); }
  public ByteBuffer outputOfOpInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 16, 1); }
  public String inputsForOp(int j) { int o = __offset(18); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int inputsForOpLength() { int o = __offset(18); return o != 0 ? __vector_len(o) : 0; }
  public StringVector inputsForOpVector() { return inputsForOpVector(new StringVector()); }
  public StringVector inputsForOpVector(StringVector obj) { int o = __offset(18); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String controlDepsForOp(int j) { int o = __offset(20); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int controlDepsForOpLength() { int o = __offset(20); return o != 0 ? __vector_len(o) : 0; }
  public StringVector controlDepsForOpVector() { return controlDepsForOpVector(new StringVector()); }
  public StringVector controlDepsForOpVector(StringVector obj) { int o = __offset(20); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String controlDepsForVar(int j) { int o = __offset(22); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int controlDepsForVarLength() { int o = __offset(22); return o != 0 ? __vector_len(o) : 0; }
  public StringVector controlDepsForVarVector() { return controlDepsForVarVector(new StringVector()); }
  public StringVector controlDepsForVarVector(StringVector obj) { int o = __offset(22); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String gradientVariable() { int o = __offset(24); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer gradientVariableAsByteBuffer() { return __vector_as_bytebuffer(24, 1); }
  public ByteBuffer gradientVariableInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 24, 1); }
  public String uiLabelExtra() { int o = __offset(26); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer uiLabelExtraAsByteBuffer() { return __vector_as_bytebuffer(26, 1); }
  public ByteBuffer uiLabelExtraInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 26, 1); }
  public org.nd4j.graph.FlatArray constantValue() { return constantValue(new org.nd4j.graph.FlatArray()); }
  public org.nd4j.graph.FlatArray constantValue(org.nd4j.graph.FlatArray obj) { int o = __offset(28); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }

  public static int createUIVariable(FlatBufferBuilder builder,
      int idOffset,
      int nameOffset,
      byte type,
      byte datatype,
      int shapeOffset,
      int controlDepsOffset,
      int outputOfOpOffset,
      int inputsForOpOffset,
      int controlDepsForOpOffset,
      int controlDepsForVarOffset,
      int gradientVariableOffset,
      int uiLabelExtraOffset,
      int constantValueOffset) {
    builder.startTable(13);
    UIVariable.addConstantValue(builder, constantValueOffset);
    UIVariable.addUiLabelExtra(builder, uiLabelExtraOffset);
    UIVariable.addGradientVariable(builder, gradientVariableOffset);
    UIVariable.addControlDepsForVar(builder, controlDepsForVarOffset);
    UIVariable.addControlDepsForOp(builder, controlDepsForOpOffset);
    UIVariable.addInputsForOp(builder, inputsForOpOffset);
    UIVariable.addOutputOfOp(builder, outputOfOpOffset);
    UIVariable.addControlDeps(builder, controlDepsOffset);
    UIVariable.addShape(builder, shapeOffset);
    UIVariable.addName(builder, nameOffset);
    UIVariable.addId(builder, idOffset);
    UIVariable.addDatatype(builder, datatype);
    UIVariable.addType(builder, type);
    return UIVariable.endUIVariable(builder);
  }

  public static void startUIVariable(FlatBufferBuilder builder) { builder.startTable(13); }
  public static void addId(FlatBufferBuilder builder, int idOffset) { builder.addOffset(0, idOffset, 0); }
  public static void addName(FlatBufferBuilder builder, int nameOffset) { builder.addOffset(1, nameOffset, 0); }
  public static void addType(FlatBufferBuilder builder, byte type) { builder.addByte(2, type, 0); }
  public static void addDatatype(FlatBufferBuilder builder, byte datatype) { builder.addByte(3, datatype, 0); }
  public static void addShape(FlatBufferBuilder builder, int shapeOffset) { builder.addOffset(4, shapeOffset, 0); }
  public static int createShapeVector(FlatBufferBuilder builder, long[] data) { builder.startVector(8, data.length, 8); for (int i = data.length - 1; i >= 0; i--) builder.addLong(data[i]); return builder.endVector(); }
  public static void startShapeVector(FlatBufferBuilder builder, int numElems) { builder.startVector(8, numElems, 8); }
  public static void addControlDeps(FlatBufferBuilder builder, int controlDepsOffset) { builder.addOffset(5, controlDepsOffset, 0); }
  public static int createControlDepsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startControlDepsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addOutputOfOp(FlatBufferBuilder builder, int outputOfOpOffset) { builder.addOffset(6, outputOfOpOffset, 0); }
  public static void addInputsForOp(FlatBufferBuilder builder, int inputsForOpOffset) { builder.addOffset(7, inputsForOpOffset, 0); }
  public static int createInputsForOpVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startInputsForOpVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addControlDepsForOp(FlatBufferBuilder builder, int controlDepsForOpOffset) { builder.addOffset(8, controlDepsForOpOffset, 0); }
  public static int createControlDepsForOpVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startControlDepsForOpVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addControlDepsForVar(FlatBufferBuilder builder, int controlDepsForVarOffset) { builder.addOffset(9, controlDepsForVarOffset, 0); }
  public static int createControlDepsForVarVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startControlDepsForVarVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addGradientVariable(FlatBufferBuilder builder, int gradientVariableOffset) { builder.addOffset(10, gradientVariableOffset, 0); }
  public static void addUiLabelExtra(FlatBufferBuilder builder, int uiLabelExtraOffset) { builder.addOffset(11, uiLabelExtraOffset, 0); }
  public static void addConstantValue(FlatBufferBuilder builder, int constantValueOffset) { builder.addOffset(12, constantValueOffset, 0); }
  public static int endUIVariable(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public UIVariable get(int j) { return get(new UIVariable(), j); }
    public UIVariable get(UIVariable obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

