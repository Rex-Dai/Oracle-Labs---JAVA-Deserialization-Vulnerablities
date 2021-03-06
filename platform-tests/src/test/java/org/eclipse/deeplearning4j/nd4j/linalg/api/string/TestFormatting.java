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

package org.eclipse.deeplearning4j.nd4j.linalg.api.string;

import lombok.extern.slf4j.Slf4j;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.string.NDArrayStrings;


import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.NDARRAY_SERDE)
public class TestFormatting extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoByTwo(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        System.out.println(new NDArrayStrings().format(arr));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNd4jArrayString(Nd4jBackend backend) {

        INDArray arr = Nd4j.create(new float[]{1f, 20000000f, 40.838383f, 3f}, new int[]{2, 2});

        String serializedData1 = new NDArrayStrings(",", 3).format(arr);
        log.info("\n" + serializedData1);
        String expected1 = "[[1.000,40.838],\n" + " [2e7,3.000]]";
        assertEquals(expected1.replaceAll(" ", ""), serializedData1.replaceAll(" ", ""));

        String serializedData2 = new NDArrayStrings().format(arr);
        log.info("\n" + serializedData2);
        String expected2 = "[[1.0000,40.8384],\n" + " [2e7,3.0000]]";
        assertEquals(expected2.replaceAll(" ", ""), serializedData2.replaceAll(" ", ""));

        String serializedData3 = new NDArrayStrings(",", "000.00##E0").format(arr);
        String expected3 = "[[100.00E-2,408.3838E-1],\n" + " [200.00E5,300.00E-2]]";
        log.info("\n"+serializedData3);
        assertEquals(expected3.replaceAll(" ", ""), serializedData3.replaceAll(" ", ""));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRange(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[][]{
                {-1,0,1,0},
                {-0.1, 0.1, -10, 10},
                {-1e-2, 1e-2, -1e2, 1e2},
                {-1e-3, 1e-3, -1e3, 1e3},
                {-1e-4, 1e-4, -1e4, 1e4},
                {-1e-8, 1e-8, -1e8, 1e8},
                {-1e-30, 1e-30, -1e30, 1e30},
                {-1e-50, 1e-50, -1e50, 1e50}, //larger than float
        });
        log.info("\n"+arr.toString());

        arr = Nd4j.create(new double[][]{
                {1.0001e4, 1e5},
                {0.11, 0.269},
        });
        arr = arr.reshape(2,2,1);
        log.info("\n"+arr.toString());

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
