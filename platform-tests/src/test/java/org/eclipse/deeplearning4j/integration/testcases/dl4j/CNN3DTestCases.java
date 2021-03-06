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

package org.eclipse.deeplearning4j.integration.testcases.dl4j;

import org.deeplearning4j.datasets.iterator.utilty.SingletonMultiDataSetIterator;
import org.eclipse.deeplearning4j.integration.ModelType;
import org.eclipse.deeplearning4j.integration.TestCase;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.eclipse.deeplearning4j.integration.TestUtils;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.primitives.Pair;

import java.util.Collections;
import java.util.List;

public class CNN3DTestCases {


    /**
     * A simple synthetic CNN 3d test case using all CNN 3d layers:
     * Subsampling, Upsampling, Convolution, Cropping, Zero padding
     */
    public static TestCase getCnn3dTestCaseSynthetic(){
        return new TestCase() {
            {
                testName = "Cnn3dSynthetic";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;
            }

            @Override
            public ModelType modelType() {
                return ModelType.MLN;
            }

            public Object getConfiguration() throws Exception {
                int nChannels = 3; // Number of input channels
                int outputNum = 10; // The number of possible outcomes
                int seed = 123;

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        .layer(new Convolution3D.Builder(3,3,3)
                                .dataFormat(Convolution3D.DataFormat.NCDHW)
                                .nIn(nChannels)
                                .stride(2, 2, 2)
                                .nOut(8)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(new Subsampling3DLayer.Builder(PoolingType.MAX)
                                .kernelSize(2, 2, 2)
                                .stride(2, 2, 2)
                                .build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutional3D(8,8,8,nChannels))
                        .build();

                return conf;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                Nd4j.getRandom().setSeed(12345);
                //NCDHW format
                INDArray arr = Nd4j.rand(new int[]{2, 3, 8, 8, 8});
                INDArray labels = TestUtils.randomOneHot(2, 10);
                return new org.nd4j.linalg.dataset.MultiDataSet(arr, labels);
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                return new SingletonMultiDataSetIterator(getGradientsTestData());
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                return getTrainingData();
            }

            @Override
            public List<Pair<INDArray[],INDArray[]>> getPredictionsTestData() throws Exception {
                MultiDataSet mds = getGradientsTestData();
                return Collections.singletonList(new Pair<>(mds.getFeatures(), null));
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{new Evaluation()};
            }

        };
    };

}
