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

/**
 * @const
 * @namespace
 */
var nd4j = nd4j || {};

/**
 * @const
 * @namespace
 */
nd4j.graph = nd4j.graph || {};

/**
 * @enum
 */
nd4j.graph.VarType = {
  VARIABLE: 0,
  CONSTANT: 1,
  ARRAY: 2,
  PLACEHOLDER: 3
};

/**
 * @constructor
 */
nd4j.graph.FlatVariable = function() {
  /**
   * @type {flatbuffers.ByteBuffer}
   */
  this.bb = null;

  /**
   * @type {number}
   */
  this.bb_pos = 0;
};

/**
 * @param {number} i
 * @param {flatbuffers.ByteBuffer} bb
 * @returns {nd4j.graph.FlatVariable}
 */
nd4j.graph.FlatVariable.prototype.__init = function(i, bb) {
  this.bb_pos = i;
  this.bb = bb;
  return this;
};

/**
 * @param {flatbuffers.ByteBuffer} bb
 * @param {nd4j.graph.FlatVariable=} obj
 * @returns {nd4j.graph.FlatVariable}
 */
nd4j.graph.FlatVariable.getRootAsFlatVariable = function(bb, obj) {
  return (obj || new nd4j.graph.FlatVariable).__init(bb.readInt32(bb.position()) + bb.position(), bb);
};

/**
 * @param {nd4j.graph.IntPair=} obj
 * @returns {nd4j.graph.IntPair|null}
 */
nd4j.graph.FlatVariable.prototype.id = function(obj) {
  var offset = this.bb.__offset(this.bb_pos, 4);
  return offset ? (obj || new nd4j.graph.IntPair).__init(this.bb.__indirect(this.bb_pos + offset), this.bb) : null;
};

/**
 * @param {flatbuffers.Encoding=} optionalEncoding
 * @returns {string|Uint8Array|null}
 */
nd4j.graph.FlatVariable.prototype.name = function(optionalEncoding) {
  var offset = this.bb.__offset(this.bb_pos, 6);
  return offset ? this.bb.__string(this.bb_pos + offset, optionalEncoding) : null;
};

/**
 * @returns {nd4j.graph.DataType}
 */
nd4j.graph.FlatVariable.prototype.dtype = function() {
  var offset = this.bb.__offset(this.bb_pos, 8);
  return offset ? /** @type {nd4j.graph.DataType} */ (this.bb.readInt8(this.bb_pos + offset)) : nd4j.graph.DataType.INHERIT;
};

/**
 * @param {number} index
 * @returns {flatbuffers.Long}
 */
nd4j.graph.FlatVariable.prototype.shape = function(index) {
  var offset = this.bb.__offset(this.bb_pos, 10);
  return offset ? this.bb.readInt64(this.bb.__vector(this.bb_pos + offset) + index * 8) : this.bb.createLong(0, 0);
};

/**
 * @returns {number}
 */
nd4j.graph.FlatVariable.prototype.shapeLength = function() {
  var offset = this.bb.__offset(this.bb_pos, 10);
  return offset ? this.bb.__vector_len(this.bb_pos + offset) : 0;
};

/**
 * @param {nd4j.graph.FlatArray=} obj
 * @returns {nd4j.graph.FlatArray|null}
 */
nd4j.graph.FlatVariable.prototype.ndarray = function(obj) {
  var offset = this.bb.__offset(this.bb_pos, 12);
  return offset ? (obj || new nd4j.graph.FlatArray).__init(this.bb.__indirect(this.bb_pos + offset), this.bb) : null;
};

/**
 * @returns {number}
 */
nd4j.graph.FlatVariable.prototype.device = function() {
  var offset = this.bb.__offset(this.bb_pos, 14);
  return offset ? this.bb.readInt32(this.bb_pos + offset) : 0;
};

/**
 * @returns {nd4j.graph.VarType}
 */
nd4j.graph.FlatVariable.prototype.variabletype = function() {
  var offset = this.bb.__offset(this.bb_pos, 16);
  return offset ? /** @type {nd4j.graph.VarType} */ (this.bb.readInt8(this.bb_pos + offset)) : nd4j.graph.VarType.VARIABLE;
};

/**
 * @param {flatbuffers.Builder} builder
 */
nd4j.graph.FlatVariable.startFlatVariable = function(builder) {
  builder.startObject(7);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {flatbuffers.Offset} idOffset
 */
nd4j.graph.FlatVariable.addId = function(builder, idOffset) {
  builder.addFieldOffset(0, idOffset, 0);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {flatbuffers.Offset} nameOffset
 */
nd4j.graph.FlatVariable.addName = function(builder, nameOffset) {
  builder.addFieldOffset(1, nameOffset, 0);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {nd4j.graph.DataType} dtype
 */
nd4j.graph.FlatVariable.addDtype = function(builder, dtype) {
  builder.addFieldInt8(2, dtype, nd4j.graph.DataType.INHERIT);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {flatbuffers.Offset} shapeOffset
 */
nd4j.graph.FlatVariable.addShape = function(builder, shapeOffset) {
  builder.addFieldOffset(3, shapeOffset, 0);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {Array.<flatbuffers.Long>} data
 * @returns {flatbuffers.Offset}
 */
nd4j.graph.FlatVariable.createShapeVector = function(builder, data) {
  builder.startVector(8, data.length, 8);
  for (var i = data.length - 1; i >= 0; i--) {
    builder.addInt64(data[i]);
  }
  return builder.endVector();
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {number} numElems
 */
nd4j.graph.FlatVariable.startShapeVector = function(builder, numElems) {
  builder.startVector(8, numElems, 8);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {flatbuffers.Offset} ndarrayOffset
 */
nd4j.graph.FlatVariable.addNdarray = function(builder, ndarrayOffset) {
  builder.addFieldOffset(4, ndarrayOffset, 0);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {number} device
 */
nd4j.graph.FlatVariable.addDevice = function(builder, device) {
  builder.addFieldInt32(5, device, 0);
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {nd4j.graph.VarType} variabletype
 */
nd4j.graph.FlatVariable.addVariabletype = function(builder, variabletype) {
  builder.addFieldInt8(6, variabletype, nd4j.graph.VarType.VARIABLE);
};

/**
 * @param {flatbuffers.Builder} builder
 * @returns {flatbuffers.Offset}
 */
nd4j.graph.FlatVariable.endFlatVariable = function(builder) {
  var offset = builder.endObject();
  return offset;
};

/**
 * @param {flatbuffers.Builder} builder
 * @param {flatbuffers.Offset} offset
 */
nd4j.graph.FlatVariable.finishFlatVariableBuffer = function(builder, offset) {
  builder.finish(offset);
};

// Exports for Node.js and RequireJS
this.nd4j = nd4j;
