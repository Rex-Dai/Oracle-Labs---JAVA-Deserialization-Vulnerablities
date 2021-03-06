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

package org.nd4j.linalg.api.ops.impl.transforms.custom;


import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class SpaceToBatch extends DynamicCustomOp {

    protected int[] blocks;
    protected int[][] padding;

    public SpaceToBatch() {
    }

    public SpaceToBatch(SameDiff sameDiff, SDVariable x, int[] blocks, int[] paddingTop, int... paddingBottom) {
        this(sameDiff, new SDVariable[]{x}, blocks, new int[][]{paddingBottom, paddingBottom}, false);
    }

    public SpaceToBatch(SameDiff sameDiff, SDVariable[] args, int[] blocks, int[][] padding, boolean inPlace) {
        super(null, sameDiff, new SDVariable[]{args[0], sameDiff.constant(Nd4j.createFromArray(padding))}, inPlace);

        this.blocks = blocks;
        this.padding = padding;

        addIArgument(blocks[0]);
    }

    public SpaceToBatch(INDArray x, int[] blocks, int[] paddingTop, int... paddingBottom) {
        addInputArgument(x);
        this.blocks = blocks;
        this.padding = padding;

        addIArgument(blocks[0]);
    }

    @Override
    public String opName() {
        return "space_to_batch";
    }

    @Override
    public String onnxName() {
        return "space_to_batch";
    }

    @Override
    public String tensorflowName() {
        return "SpaceToBatch";
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new HashMap<>();
        if(blocks != null)
            ret.put("blocks",blocks);
        if(padding != null)
            ret.put("padding",padding);
        return ret;
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("padding")) {
            int[][] padding = (int[][]) properties.get("padding");
            this.padding =  padding;
        }
        if(properties.containsKey("blocks")) {
            int[] blocks = (int[]) properties.get("blocks");
            this.blocks = blocks;
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Inverse of space to batch is batch to space with same blocks and crops as padding
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        return Arrays.asList(sameDiff.cnn().batchToSpace(gradient, blocks, padding[0], padding[1]));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        return Collections.singletonList(dataTypes.get(0));
    }

}
