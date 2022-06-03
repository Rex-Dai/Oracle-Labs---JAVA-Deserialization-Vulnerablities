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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;


/**
 * Pooling2D operation
 */
@Slf4j
@Getter
public class Pooling2D extends DynamicCustomOp {

    protected Pooling2DConfig config;

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }
    
    /**
     * Divisor mode for average pooling only. 3 modes are supported:
     * MODE_0:
     * EXCLUDE_PADDING:
     * INCLUDE_PADDING: Always do sum(window) / (kH*kW) even if padding is present.
     */
    public enum Divisor {
        EXCLUDE_PADDING, INCLUDE_PADDING
    }

    public Pooling2D() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    @SuppressWarnings("Used in lombok")
    public Pooling2D(SameDiff sameDiff, SDVariable[] inputs,
            Pooling2DConfig config) {
        super(null, sameDiff, inputs, false);

        this.config = config;
        addArgs();
    }

    public Pooling2D(@NonNull INDArray[] inputs, INDArray[] outputs, @NonNull Pooling2DConfig config){
        super(inputs, outputs);

        this.config = config;
        addArgs();
    }

    public Pooling2D(@NonNull INDArray input, INDArray output, @NonNull Pooling2DConfig config){
        super(new INDArray[]{input}, wrapOrNull(output));

        this.config = config;
        addArgs();
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    private void addArgs() {
        val t = config.getType();

        addIArgument(config.getKH());
        addIArgument(config.getKW());
        addIArgument(config.getSH());
        addIArgument(config.getSW());
        addIArgument(config.getPH());
        addIArgument(config.getPW());
        addIArgument(config.getDH());
        addIArgument(config.getDW());
        addIArgument(config.getPaddingMode().index);
        addIArgument((t == Pooling2DType.AVG) ? config.getDivisor().ordinal() : (int)config.getExtra());
        addIArgument(ArrayUtil.fromBoolean(config.isNHWC()));
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool2d";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
       if(!iArguments.isEmpty() && config == null) {
           LinAlgExceptions.assertAllConfigured(this,9);
           createConfigFromArgs();
       }

        Pooling2DDerivative pooling2DDerivative = Pooling2DDerivative.derivativeBuilder()
                .inputs(inputs.toArray(new SDVariable[inputs.size()]))
                .sameDiff(sameDiff)
                .config(config)
                .build();
        ret.addAll(Arrays.asList(pooling2DDerivative.outputVariables()));
        return ret;
    }

    private void createConfigFromArgs() {
        config = Pooling2DConfig.builder()
                .kH(iArguments.get(0))
                .kW(iArguments.get(1))
                .sH(iArguments.get(2))
                .sW(iArguments.get(3))
                .pH(iArguments.get(4))
                .pW(iArguments.get(5))
                .dH(iArguments.get(6))
                .dW(iArguments.get(7))
                .paddingMode(PaddingMode.fromNumber(iArguments.get(8).intValue()))
                .type(null)
                .build();
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sH = tfStrides.get(1);
        val sW = tfStrides.get(2);

        val aKernels = nodeDef.getAttrOrThrow("ksize");
        val tfKernels = aKernels.getList().getIList();

        val kH = tfKernels.get(1);
        val kW = tfKernels.get(2);

        val aPadding = nodeDef.getAttrOrThrow("padding");
        val padding = aPadding.getList().getIList();

        val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"","");

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sH(sH.intValue())
                .sW(sW.intValue())
                .type(null)
                .paddingMode(PaddingMode.valueOf(paddingMode))
                .kH(kH.intValue())
                .kW(kW.intValue())
                .pH(padding.get(0).intValue())
                .pW(padding.get(1).intValue())
                .build();
        this.config = pooling2DConfig;
        addArgs();
        log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kH, kW, sH, sW, aPadding);


    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        val paddingMode = attributesForNode.get("auto_pad").getS().toStringUtf8();
        val kernelShape = attributesForNode.get("kernel_shape").getIntsList();
        val padding = attributesForNode.get("pads").getIntsList();
        val strides = attributesForNode.get("strides").getIntsList();

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .sW(strides.get(0).intValue())
                .sH(strides.get(1).intValue())
                .type(null)
                .paddingMode(PaddingMode.valueOf(paddingMode))
                .kH(kernelShape.get(0).intValue())
                .kW(kernelShape.get(1).intValue())
                .pH(padding.get(0).intValue())
                .pW(padding.get(1).intValue())
                .build();
        this.config = pooling2DConfig;
        addArgs();
    }


    public String getPoolingPrefix() {
        if (config == null)
            return "somepooling";

        switch(config.getType()) {
            case AVG:return "avg";
            case MAX: return "max";
            case PNORM: return "pnorm";
            default: throw new IllegalStateException("No pooling type found.");
        }
    }


    @Override
    public String onnxName() {
        return "Pooling";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 1, "Expected 1 input data type for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
