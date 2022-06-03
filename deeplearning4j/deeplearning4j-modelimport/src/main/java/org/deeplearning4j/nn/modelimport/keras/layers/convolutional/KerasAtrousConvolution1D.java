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

package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;

public class KerasAtrousConvolution1D extends KerasConvolution {

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasAtrousConvolution1D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasAtrousConvolution1D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasAtrousConvolution1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        hasBias = KerasLayerUtils.getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;

        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);

        IWeightInit init = KerasInitilizationUtils.getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);

        Convolution1DLayer.Builder builder = new Convolution1DLayer.Builder().name(this.layerName)
                .nOut(KerasLayerUtils.getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(KerasActivationUtils.getIActivationFromConfig(layerConfig, conf))
                .weightInit(init)
                .dilation(getDilationRate(layerConfig, 1, conf, true)[0])
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 1, conf, kerasMajorVersion)[0])
                .hasBias(hasBias)
                .rnnDataFormat(dimOrder == KerasLayer.DimOrder.TENSORFLOW ? RNNFormat.NWC : RNNFormat.NCW)
                .stride(getStrideFromConfig(layerConfig, 1, conf)[0]);
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 1, conf, kerasMajorVersion);
        if (hasBias)
            builder.biasInit(0.0);
        if (padding != null)
            builder.padding(padding[0]);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (weightConstraint != null)
            builder.constrainWeights(weightConstraint);
        this.layer = builder.build();
        Convolution1DLayer convolution1DLayer = (Convolution1DLayer) layer;
        convolution1DLayer.setDefaultValueOverriden(true);
    }

    /**
     * Get DL4J ConvolutionLayer.
     *
     * @return ConvolutionLayer
     */
    public Convolution1DLayer getAtrousConvolution1D() {
        return (Convolution1DLayer) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Convolution layer accepts only one input (received " + inputType.length + ")");
        return this.getAtrousConvolution1D().getOutputType(-1, inputType[0]);
    }
}