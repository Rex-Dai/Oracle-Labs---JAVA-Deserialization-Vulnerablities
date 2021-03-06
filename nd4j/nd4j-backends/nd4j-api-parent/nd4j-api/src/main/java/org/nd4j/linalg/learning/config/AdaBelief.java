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

package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaBeliefUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Map;

/**
 * AdaBelief
 * https://arxiv.org/pdf/2010.07468.pdf
 */
@Data
@Builder(builderClassName = "Builder")
public class AdaBelief implements IUpdater {

    public static final double DEFAULT_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_EPSILON = 1e-14;
    public static final double DEFAULT_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_BETA2_VAR_DECAY = 0.999;

    @lombok.Builder.Default private double learningRate = DEFAULT_LEARNING_RATE; // learning rate
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double beta1 = DEFAULT_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    @lombok.Builder.Default private double beta2 = DEFAULT_BETA2_VAR_DECAY; // gradient sqrt decay rate
    @lombok.Builder.Default private double epsilon = DEFAULT_EPSILON;

    public AdaBelief() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_BETA1_MEAN_DECAY, DEFAULT_BETA2_VAR_DECAY,
                        DEFAULT_EPSILON);
    }

    public AdaBelief(double learningRate) {
        this(learningRate, null, DEFAULT_BETA1_MEAN_DECAY, DEFAULT_BETA2_VAR_DECAY, DEFAULT_EPSILON);
    }

    public AdaBelief(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_BETA1_MEAN_DECAY, DEFAULT_BETA2_VAR_DECAY, DEFAULT_EPSILON);
    }

    public AdaBelief(double learningRate, double beta1, double beta2, double epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    private AdaBelief(@JsonProperty("learningRate") double learningRate,
                 @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                 @JsonProperty("beta1") double beta1,
                 @JsonProperty("beta2") double beta2,
                 @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return 2 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        AdaBeliefUpdater u = new AdaBeliefUpdater(this);
        viewArray = viewArray.reshape(viewArray.length());
        long[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 2;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public GradientUpdater instantiate(Map<String, INDArray> updaterState, boolean initializeStateArrays) {
        AdaBeliefUpdater u = new AdaBeliefUpdater(this);
        u.setState(updaterState, initializeStateArrays);
        return u;
    }

    @Override
    public AdaBelief clone() {
        return new AdaBelief(learningRate, learningRateSchedule, beta1, beta2, epsilon);
    }

    @Override
    public double getLearningRate(int iteration, int epoch){
        if(learningRateSchedule != null){
            return learningRateSchedule.valueAt(iteration, epoch);
        }
        return learningRate;
    }

    @Override
    public boolean hasLearningRate() {
        return true;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
        this.learningRate = lr;
        this.learningRateSchedule = lrSchedule;
    }

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
