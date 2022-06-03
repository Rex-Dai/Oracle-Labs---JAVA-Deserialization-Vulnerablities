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

package org.datavec.spark.transform.filter;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.writable.Writable;

import java.util.List;

@AllArgsConstructor
public class SparkFilterFunction implements Function<List<Writable>, Boolean> {

    private final Filter filter;

    @Override
    public Boolean call(List<Writable> v1) throws Exception {
        return !filter.removeExample(v1); //Spark: return true to keep example (Filter: return true to remove)
    }
}
