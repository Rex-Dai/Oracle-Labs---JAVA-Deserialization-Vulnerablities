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


package org.nd4j.linalg.aurora;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.BasicMemoryManager;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;

/**
 * @author Adam Gibson
 */
@Slf4j
public class AuroraMemoryManager extends BasicMemoryManager {
    /**
     * This method returns
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        Pointer ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().mallocDevice(bytes, -1, 0);

        if (ptr == null || ptr.address() == 0L)
            throw new OutOfMemoryError("Failed to allocate [" + bytes + "] bytes");

        //log.info("Allocating {} bytes at MemoryManager", bytes);

        if (initialize)
            NativeOpsHolder.getInstance().getDeviceNativeOps().memsetSync(ptr, 0, bytes, 0, null);

        return ptr;
    }

    /**
     * This method releases previously allocated memory chunk
     *
     * @param pointer
     * @param kind
     * @return
     */
    @Override
    public void release(@NonNull Pointer pointer, MemoryKind kind) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().freeDevice(pointer, -1);
        pointer.setNull();
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {
        super.collect(arrays);
    }

    /**
     * Nd4j-native backend doesn't use periodic GC. This method will always return false.
     *
     * @return
     */
    @Override
    public boolean isPeriodicGcActive() {
        return false;
    }

    @Override
    public void memcpy(DataBuffer dstBuffer, DataBuffer srcBuffer) {
        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpySync(dstBuffer.addressPointer(), srcBuffer.addressPointer(),
                        srcBuffer.length() * srcBuffer.getElementSize(), 3, null);

        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfD, srcBuffer.length() * srcBuffer.getElementSize(), MemcpyDirection.HOST_TO_HOST);
    }

    @Override
    public void memset(INDArray array) {
        if (array.isView()) {
            array.assign(0.0);
            return;
        }

        NativeOpsHolder.getInstance().getDeviceNativeOps().memsetSync(new PagedPointer(array.data().addressPointer(), 0), 0, array.data().length() * Nd4j.sizeOfDataType(array.data().dataType()), 0, null);
    }

    @Override
    public Map<Integer, Long> getBandwidthUse() {
        return null;
    }

    @Override
    public long allocatedMemory(Integer deviceId) {
        return Pointer.totalBytes() + AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.GENERAL, deviceId) + AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.WORKSPACE, deviceId);
    }
}
