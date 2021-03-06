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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;

/**
 * @author Max Pumperla
 */
@DisplayName("Upsampling 1 D Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class Upsampling1DTest extends BaseDL4JTest {

    private int nExamples = 1;

    private int depth = 20;

    private int nChannelsIn = 1;

    private int inputLength = 28;

    private int size = 2;

    private int outputLength = inputLength * size;

    private INDArray epsilon = Nd4j.ones(nExamples, depth, outputLength);

    @Test
    @DisplayName("Test Upsampling 1 D")
    void testUpsampling1D() throws Exception {
        double[] outArray = new double[] { 1., 1., 2., 2., 3., 3., 4., 4. };
        INDArray containedExpectedOut = Nd4j.create(outArray, new int[] { 1, 1, 8 });
        INDArray containedInput = getContainedData();
        INDArray input = getData();
        Layer layer = getUpsampling1DLayer();
        INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);
        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(new long[] { nExamples, nChannelsIn, outputLength }, output.shape()));
        assertEquals(nChannelsIn, output.size(1), 1e-4);
    }

    @Test
    @DisplayName("Test Upsampling 1 D Backprop")
    void testUpsampling1DBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput = Nd4j.create(new double[] { 1., 3., 2., 6., 7., 2., 5., 5. }, new int[] { 1, 1, 8 });
        INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] { 4., 8., 9., 10. }, new int[] { 1, 1, 4 });
        INDArray input = getContainedData();
        Layer layer = getUpsampling1DLayer();
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.getSecond().shape().length);
        INDArray input2 = getData();
        layer.activate(input2, false, LayerWorkspaceMgr.noWorkspaces());
        val depth = input2.size(1);
        epsilon = Nd4j.ones(5, depth, outputLength);
        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1));
    }

    private Layer getUpsampling1DLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123).layer(new Upsampling1D.Builder(size).build()).build();
        return conf.getLayer().instantiate(conf, null, 0, null, true, Nd4j.defaultFloatingPointType());
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5, 5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        INDArray features = mnist.getFeatures().reshape(nExamples, nChannelsIn, inputLength, inputLength);
        return features.slice(0, 3);
    }

    private INDArray getContainedData() {
        INDArray ret = Nd4j.create(new double[] { 1., 2., 3., 4. }, new int[] { 1, 1, 4 });
        return ret;
    }
}
