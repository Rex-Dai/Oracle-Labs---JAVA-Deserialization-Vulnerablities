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
public final class UIOp extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static UIOp getRootAsUIOp(ByteBuffer _bb) { return getRootAsUIOp(_bb, new UIOp()); }
  public static UIOp getRootAsUIOp(ByteBuffer _bb, UIOp obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public UIOp __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public String name() { int o = __offset(4); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer nameAsByteBuffer() { return __vector_as_bytebuffer(4, 1); }
  public ByteBuffer nameInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 4, 1); }
  public String opName() { int o = __offset(6); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer opNameAsByteBuffer() { return __vector_as_bytebuffer(6, 1); }
  public ByteBuffer opNameInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 6, 1); }
  public String inputs(int j) { int o = __offset(8); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int inputsLength() { int o = __offset(8); return o != 0 ? __vector_len(o) : 0; }
  public StringVector inputsVector() { return inputsVector(new StringVector()); }
  public StringVector inputsVector(StringVector obj) { int o = __offset(8); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String outputs(int j) { int o = __offset(10); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int outputsLength() { int o = __offset(10); return o != 0 ? __vector_len(o) : 0; }
  public StringVector outputsVector() { return outputsVector(new StringVector()); }
  public StringVector outputsVector(StringVector obj) { int o = __offset(10); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String controlDeps(int j) { int o = __offset(12); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int controlDepsLength() { int o = __offset(12); return o != 0 ? __vector_len(o) : 0; }
  public StringVector controlDepsVector() { return controlDepsVector(new StringVector()); }
  public StringVector controlDepsVector(StringVector obj) { int o = __offset(12); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }
  public String uiLabelExtra() { int o = __offset(14); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer uiLabelExtraAsByteBuffer() { return __vector_as_bytebuffer(14, 1); }
  public ByteBuffer uiLabelExtraInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 14, 1); }

  public static int createUIOp(FlatBufferBuilder builder,
      int nameOffset,
      int opNameOffset,
      int inputsOffset,
      int outputsOffset,
      int controlDepsOffset,
      int uiLabelExtraOffset) {
    builder.startTable(6);
    UIOp.addUiLabelExtra(builder, uiLabelExtraOffset);
    UIOp.addControlDeps(builder, controlDepsOffset);
    UIOp.addOutputs(builder, outputsOffset);
    UIOp.addInputs(builder, inputsOffset);
    UIOp.addOpName(builder, opNameOffset);
    UIOp.addName(builder, nameOffset);
    return UIOp.endUIOp(builder);
  }

  public static void startUIOp(FlatBufferBuilder builder) { builder.startTable(6); }
  public static void addName(FlatBufferBuilder builder, int nameOffset) { builder.addOffset(0, nameOffset, 0); }
  public static void addOpName(FlatBufferBuilder builder, int opNameOffset) { builder.addOffset(1, opNameOffset, 0); }
  public static void addInputs(FlatBufferBuilder builder, int inputsOffset) { builder.addOffset(2, inputsOffset, 0); }
  public static int createInputsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startInputsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addOutputs(FlatBufferBuilder builder, int outputsOffset) { builder.addOffset(3, outputsOffset, 0); }
  public static int createOutputsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startOutputsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addControlDeps(FlatBufferBuilder builder, int controlDepsOffset) { builder.addOffset(4, controlDepsOffset, 0); }
  public static int createControlDepsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startControlDepsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addUiLabelExtra(FlatBufferBuilder builder, int uiLabelExtraOffset) { builder.addOffset(5, uiLabelExtraOffset, 0); }
  public static int endUIOp(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public UIOp get(int j) { return get(new UIOp(), j); }
    public UIOp get(UIOp obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

