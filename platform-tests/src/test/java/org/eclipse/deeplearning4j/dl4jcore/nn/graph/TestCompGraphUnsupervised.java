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

package org.eclipse.deeplearning4j.dl4jcore.nn.graph;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Adam;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

@NativeTag
@Tag(TagNames.DL4J_OLD_API)
@Tag(TagNames.LONG_TEST)
@Tag(TagNames.LARGE_RESOURCES)
public class TestCompGraphUnsupervised extends BaseDL4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Test
    public void testVAE() throws Exception {

        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .updater(new Adam(1e-3))
                    .weightInit(WeightInit.XAVIER)
                    .inferenceWorkspaceMode(wsm)
                    .trainingWorkspaceMode(wsm)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("vae1", new VariationalAutoencoder.Builder()
                            .nIn(784)
                            .nOut(32)
                            .encoderLayerSizes(16)
                            .decoderLayerSizes(16)
                            .activation(Activation.TANH)
                            .pzxActivationFunction(Activation.SIGMOID)
                            .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                            .build(), "in")
                    .addLayer("vae2", new VariationalAutoencoder.Builder()
                            .nIn(32)
                            .nOut(8)
                            .encoderLayerSizes(16)
                            .decoderLayerSizes(16)
                            .activation(Activation.TANH)
                            .pzxActivationFunction(Activation.SIGMOID)
                            .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
                            .build(), "vae1")
                    .setOutputs("vae2")
                    .build();

            ComputationGraph cg = new ComputationGraph(conf);
            cg.init();

            DataSetIterator ds = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(8, true, 12345), 3);

            Map<String,INDArray> paramsBefore = new HashMap<>();

            //Pretrain first layer
            for(Map.Entry<String,INDArray> e : cg.paramTable().entrySet()){
                paramsBefore.put(e.getKey(), e.getValue().dup());
            }
            cg.pretrainLayer("vae1", ds);
            for(Map.Entry<String,INDArray> e : cg.paramTable().entrySet()){
                if(e.getKey().startsWith("vae1")){
                    assertNotEquals(paramsBefore.get(e.getKey()), e.getValue());
                } else {
                    assertEquals(paramsBefore.get(e.getKey()), e.getValue());
                }
            }

            int count = Nd4j.getExecutioner().exec(new MatchCondition(cg.params(), Conditions.isNan())).getInt(0);
            assertEquals(0, count);


            //Pretrain second layer
            for(Map.Entry<String,INDArray> e : cg.paramTable().entrySet()){
                paramsBefore.put(e.getKey(), e.getValue().dup());
            }
            cg.pretrainLayer("vae2", ds);
            for(Map.Entry<String,INDArray> e : cg.paramTable().entrySet()){
                if(e.getKey().startsWith("vae2")){
                    assertNotEquals(paramsBefore.get(e.getKey()), e.getValue());
                } else {
                    assertEquals(paramsBefore.get(e.getKey()), e.getValue());
                }
            }

            count = Nd4j.getExecutioner().exec(new MatchCondition(cg.params(), Conditions.isNan())).getInt(0);
            assertEquals(0, count);
        }
    }

    @Test
    public void compareImplementations() throws Exception {

        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {

            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .updater(new Adam(1e-3))
                    .weightInit(WeightInit.XAVIER)
                    .inferenceWorkspaceMode(wsm)
                    .trainingWorkspaceMode(wsm)
                    .list()
                    .layer(new VariationalAutoencoder.Builder()
                            .nIn(784)
                            .nOut(32)
                            .encoderLayerSizes(16)
                            .decoderLayerSizes(16)
                            .activation(Activation.TANH)
                            .pzxActivationFunction(Activation.SIGMOID)
                            .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                            .build())
                    .layer(new VariationalAutoencoder.Builder()
                            .nIn(32)
                            .nOut(8)
                            .encoderLayerSizes(16)
                            .decoderLayerSizes(16)
                            .activation(Activation.TANH)
                            .pzxActivationFunction(Activation.SIGMOID)
                            .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
                            .build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf2);
            net.init();

            ComputationGraph cg = net.toComputationGraph();
            cg.getConfiguration().setInferenceWorkspaceMode(wsm);
            cg.getConfiguration().setTrainingWorkspaceMode(wsm);
            DataSetIterator ds = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(1, true, 12345), 1);
            Nd4j.getRandom().setSeed(12345);
            net.pretrainLayer(0, ds);

            ds = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(1, true, 12345), 1);
            Nd4j.getRandom().setSeed(12345);
            cg.pretrainLayer("0", ds);

            assertEquals(net.params(), cg.params());
        }
    }

}
