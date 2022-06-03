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

package org.deeplearning4j.spark.parameterserver.callbacks;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.MultiDataSet;

import java.io.DataInputStream;

public class MultiDataSetDeserializationCallback implements PortableDataStreamMDSCallback {

    @Override
    public MultiDataSet compute(PortableDataStream pds) {
        try (DataInputStream is = pds.open()) {
            // TODO: do something better here
            MultiDataSet ds = new MultiDataSet();
            ds.load(is);
            return ds;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
