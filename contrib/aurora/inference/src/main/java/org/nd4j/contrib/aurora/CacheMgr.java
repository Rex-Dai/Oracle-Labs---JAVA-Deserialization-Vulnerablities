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


package org.nd4j.contrib.aurora;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

import org.nd4j.autodiff.samediff.internal.memory.AbstractMemoryMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

public class CacheMgr extends AbstractMemoryMgr {


    private final static int allowExtras = 5;

    HashMap<String, Queue<INDArray>> arrayReuse = new HashMap<>();

    String getKey(INDArray arr) {
        return Arrays.toString(arr.shape()) + "_" + arr.dataType().toString();
    }

    String getKey(DataType dataType, long... shape) {
        return Arrays.toString(shape) + "_" + dataType.toString();
    }

    public CacheMgr() {
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        String key = getKey(dataType, shape);
        if (arrayReuse.containsKey(key)) {
            INDArray w = arrayReuse.get(key).poll();
            if (w != null) {
                ((BaseNDArray) w).assignNewId();
                // System.out.println("cache1:::"+key+" "+w.getId());
                return w;
            }
        }

        // Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        if (descriptor.isEmpty()) {
            INDArray ret = Nd4j.create(descriptor);
            if (detached) {
                ret = ret.detach();
            }

            return ret;
        }

        DataType dataType = descriptor.dataType();
        long[] shape = descriptor.getShape();

        String key = getKey(dataType, shape);
        if (arrayReuse.containsKey(key)) {
            INDArray w = arrayReuse.get(key).poll();
            if (w != null) {
                ((BaseNDArray) w).assignNewId();
                // System.out.println("cache1:::"+key+" "+w.getId());
                return w;
            }
        }

        // Allocation failed, allocate new array
        return Nd4j.createUninitializedDetached(dataType, shape);
    }

    @Override
    public void release(INDArray array) {
        String key = getKey(array);
        //// System.out.println(":::"+key+" "+array.getId());
        if (arrayReuse.containsKey(key)) {
            // we already have additional one for potential reuse
            Queue<INDArray> queue = arrayReuse.get(key);
            // see we have a room
            if (allowExtras > queue.size()) {
                queue.add(array);
            } else {
                // we should close it as we dont want to store to easy on space
                if (array.closeable()) {
                    array.close();
                    return;
                }
            }
        } else {
            // add it
            Queue<INDArray> queue = new LinkedList<INDArray>();
            queue.add(array);
            arrayReuse.put(key, queue);
        }
    }

    @Override
    public void close() {
        for (Queue<INDArray> as : arrayReuse.values()) {
            as.forEach(
                    w -> {
                        if (w.closeable())
                            w.close();
                    });
        }
    }

}
