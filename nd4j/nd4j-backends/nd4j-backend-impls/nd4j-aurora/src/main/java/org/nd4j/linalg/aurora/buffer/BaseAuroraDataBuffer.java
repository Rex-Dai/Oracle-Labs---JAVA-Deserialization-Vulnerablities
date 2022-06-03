/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.aurora.buffer;

import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.nio.ByteBuffer;

import static org.nd4j.linalg.api.buffer.DataType.INT8;

/**
 * Base implementation for DataBuffer for CPU-like backend
 *
 * @author Adam Gibson
 */
public abstract class BaseAuroraDataBuffer extends BaseDataBuffer implements Deallocatable {
    private static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    protected transient OpaqueDataBuffer ptrDataBuffer;

    private transient final long instanceId = Nd4j.getDeallocatorService().nextValue();

    protected BaseAuroraDataBuffer() {

    }


    @Override
    public String getUniqueId() {
        return new String("BCDB_" + instanceId);
    }

    @Override
    public Deallocator deallocator() {
        return new AuroraDeallocator(this);
    }

    public OpaqueDataBuffer getOpaqueDataBuffer() {
        if (released)
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        return ptrDataBuffer;
    }

    @Override
    public int targetDevice() {
        // TODO: once we add NUMA support this might change. Or might not.
        return 0;
    }

    static class DeviceByteIndexer extends ByteRawIndexer {
        BytePointer temp = new BytePointer(1);

        DeviceByteIndexer(BytePointer pointer) {
            super(pointer);
        }

        @Override
        public byte get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i), 1, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public ByteIndexer put(long i, byte b) {
            temp.put(b);
            nativeOps.memcpySync(pointer.position(i), temp, 1, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceUByteIndexer extends UByteRawIndexer {
        BytePointer temp = new BytePointer(1);

        DeviceUByteIndexer(BytePointer pointer) {
            super(pointer);
        }

        @Override
        public int get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i), 1, 2, null);
            pointer.position(0);
            return temp.get() & 0xFF;
        }

        @Override
        public UByteIndexer put(long i, int b) {
            temp.put((byte)b);
            nativeOps.memcpySync(pointer.position(i), temp, 1, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceShortIndexer extends ShortRawIndexer {
        ShortPointer temp = new ShortPointer(1);

        DeviceShortIndexer(ShortPointer pointer) {
            super(pointer);
        }

        @Override
        public short get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 2), 2, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public ShortIndexer put(long i, short s) {
            temp.put(s);
            nativeOps.memcpySync(pointer.position(i * 2), temp, 2, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceUShortIndexer extends UShortRawIndexer {
        ShortPointer temp = new ShortPointer(1);

        DeviceUShortIndexer(ShortPointer pointer) {
            super(pointer);
        }

        @Override
        public int get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 2), 2, 2, null);
            pointer.position(0);
            return temp.get() & 0xFFFF;
        }

        @Override
        public UShortIndexer put(long i, int s) {
            temp.put((short)s);
            nativeOps.memcpySync(pointer.position(i * 2), temp, 2, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceIntIndexer extends IntRawIndexer {
        IntPointer temp = new IntPointer(1);

        DeviceIntIndexer(IntPointer pointer) {
            super(pointer);
        }

        @Override
        public int get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 4), 4, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public IntIndexer put(long i, int n) {
            temp.put(n);
            nativeOps.memcpySync(pointer.position(i * 4), temp, 4, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceUIntIndexer extends UIntRawIndexer {
        IntPointer temp = new IntPointer(1);

        DeviceUIntIndexer(IntPointer pointer) {
            super(pointer);
        }

        @Override
        public long get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 4), 4, 2, null);
            pointer.position(0);
            return temp.get() & 0xFFFFFFFFL;
        }

        @Override
        public UIntIndexer put(long i, long n) {
            temp.put((int)n);
            nativeOps.memcpySync(pointer.position(i * 4), temp, 4, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceLongIndexer extends LongRawIndexer {
        LongPointer temp = new LongPointer(1);

        DeviceLongIndexer(LongPointer pointer) {
            super(pointer);
        }

        @Override
        public long get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 8), 8, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public LongIndexer put(long i, long n) {
            temp.put(n);
            nativeOps.memcpySync(pointer.position(i * 8), temp, 8, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceFloatIndexer extends FloatRawIndexer {
        FloatPointer temp = new FloatPointer(1);

        DeviceFloatIndexer(FloatPointer pointer) {
            super(pointer);
        }

        @Override
        public float get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 4), 4, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public FloatIndexer put(long i, float f) {
            temp.put(f);
            nativeOps.memcpySync(pointer.position(i * 4), temp, 4, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceDoubleIndexer extends DoubleRawIndexer {
        DoublePointer temp = new DoublePointer(1);

        DeviceDoubleIndexer(DoublePointer pointer) {
            super(pointer);
        }

        @Override
        public double get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 8), 8, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public DoubleIndexer put(long i, double d) {
            temp.put(d);
            nativeOps.memcpySync(pointer.position(i * 8), temp, 8, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceBooleanIndexer extends BooleanRawIndexer {
        BooleanPointer temp = new BooleanPointer(1);

        DeviceBooleanIndexer(BooleanPointer pointer) {
            super(pointer);
        }

        @Override
        public boolean get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i), 1, 2, null);
            pointer.position(0);
            return temp.get();
        }

        @Override
        public BooleanIndexer put(long i, boolean b) {
            temp.put(b);
            nativeOps.memcpySync(pointer.position(i), temp, 1, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceHalfIndexer extends HalfRawIndexer {
        ShortPointer temp = new ShortPointer(1);

        DeviceHalfIndexer(ShortPointer pointer) {
            super(pointer);
        }

        @Override
        public float get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 2), 2, 2, null);
            pointer.position(0);
            return toFloat(temp.get());
        }

        @Override
        public HalfIndexer put(long i, float f) {
            temp.put((short)fromFloat(f));
            nativeOps.memcpySync(pointer.position(i * 2), temp, 2, 1, null);
            pointer.position(0);
            return this;
        }
    }

    static class DeviceBfloat16Indexer extends Bfloat16RawIndexer {
        ShortPointer temp = new ShortPointer(1);

        DeviceBfloat16Indexer(ShortPointer pointer) {
            super(pointer);
        }

        @Override
        public float get(long i) {
            nativeOps.memcpySync(temp, pointer.position(i * 2), 2, 2, null);
            pointer.position(0);
            return toFloat(temp.get());
        }

        @Override
        public Bfloat16Indexer put(long i, float f) {
            temp.put((short)fromFloat(f));
            nativeOps.memcpySync(pointer.position(i * 2), temp, 2, 1, null);
            pointer.position(0);
            return this;
        }
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseAuroraDataBuffer(long length, int elementSize) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        this.elementSize = (byte) elementSize;

        if (dataType() != DataType.UTF8)
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, dataType(), false);

        if (dataType() == DataType.DOUBLE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asDoublePointer();

            indexer = new DeviceDoubleIndexer((DoublePointer) pointer);
        } else if (dataType() == DataType.FLOAT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asFloatPointer();

            setIndexer(new DeviceFloatIndexer((FloatPointer) pointer));
        } else if (dataType() == DataType.INT32) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();

            setIndexer(new DeviceIntIndexer((IntPointer) pointer));
        } else if (dataType() == DataType.LONG) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();

            setIndexer(new DeviceLongIndexer((LongPointer) pointer));
        } else if (dataType() == DataType.SHORT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(new DeviceShortIndexer((ShortPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(new DeviceByteIndexer((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(new DeviceUByteIndexer((BytePointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, INT8, false);
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(new DeviceByteIndexer((BytePointer) pointer));
        } else if(dataType() == DataType.FLOAT16){
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();
            setIndexer(new DeviceHalfIndexer((ShortPointer) pointer));
        } else if(dataType() == DataType.BFLOAT16){
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();
            setIndexer(new DeviceBfloat16Indexer((ShortPointer) pointer));
        } else if(dataType() == DataType.BOOL){
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBoolPointer();
            setIndexer(new DeviceBooleanIndexer((BooleanPointer) pointer));
        } else if(dataType() == DataType.UINT16){
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();
            setIndexer(new DeviceUShortIndexer((ShortPointer) pointer));
        } else if(dataType() == DataType.UINT32){
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();
            setIndexer(new DeviceUIntIndexer((IntPointer) pointer));
        } else if (dataType() == DataType.UINT64) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();
            setIndexer(new DeviceLongIndexer((LongPointer) pointer));
        }

        Nd4j.getDeallocatorService().pickObject(this);
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseAuroraDataBuffer(int length, int elementSize, long offset) {
        this(length, elementSize);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = length - offset;
        this.underlyingLength = length;
    }


    protected BaseAuroraDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);

        // for vew we need "externally managed" pointer and deallocator registration
        ptrDataBuffer = ((BaseAuroraDataBuffer) underlyingBuffer).ptrDataBuffer.createView(length * underlyingBuffer.getElementSize(), offset * underlyingBuffer.getElementSize());
        Nd4j.getDeallocatorService().pickObject(this);


        // update pointer now
        actualizePointerAndIndexer();
    }

    protected BaseAuroraDataBuffer(ByteBuffer buffer, DataType dtype, long length, long offset) {
        this(length, Nd4j.sizeOfDataType(dtype));

        Pointer temp = null;

        switch (dataType()){
            case DOUBLE:
                temp = new DoublePointer(buffer.asDoubleBuffer());
                break;
            case FLOAT:
                temp = new FloatPointer(buffer.asFloatBuffer());
                break;
            case HALF:
                temp = new ShortPointer(buffer.asShortBuffer());
                break;
            case LONG:
                temp = new LongPointer(buffer.asLongBuffer());
                break;
            case INT:
                temp = new IntPointer(buffer.asIntBuffer());
                break;
            case SHORT:
                temp = new ShortPointer(buffer.asShortBuffer());
                break;
            case UBYTE: //Fall through
            case BYTE:
                temp = new BytePointer(buffer);
                break;
            case BOOL:
                temp = new BooleanPointer(length());
                break;
            case UTF8:
                temp = new BytePointer(length());
                break;
            case BFLOAT16:
                temp = new ShortPointer(length());
                break;
            case UINT16:
                temp = new ShortPointer(length());
                break;
            case UINT32:
                temp = new IntPointer(length());
                break;
            case UINT64:
                temp = new LongPointer(length());
                break;
        }

        val ptr = ptrDataBuffer.primaryBuffer();

        if (offset > 0)
            temp = new PagedPointer(temp.address() + offset * getElementSize());

        nativeOps.memcpySync(ptr, temp, length * Nd4j.sizeOfDataType(dtype), 3, null);
    }

    @Override
    protected double getDoubleUnsynced(long index) {
        return super.getDouble(index);
    }

    @Override
    protected float getFloatUnsynced(long index) {
        return super.getFloat(index);
    }

    @Override
    protected long getLongUnsynced(long index) {
        return super.getLong(index);
    }

    @Override
    protected int getIntUnsynced(long index) {
        return super.getInt(index);
    }

    @Override
    public void pointerIndexerByCurrentType(DataType currentType) {

        type = currentType;

        if (ptrDataBuffer == null) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length(), type, false);
            Nd4j.getDeallocatorService().pickObject(this);
        }

        actualizePointerAndIndexer();
    }

    @Override
    protected ByteBuffer wrappedBuffer() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void fillPointerWithZero() {
        nativeOps.memsetSync(ptrDataBuffer.primaryBuffer(), 0, getElementSize() * length(), 0, null);
    }

    /**
     * Instantiate a buffer with the given length
     *
     * @param length the length of the buffer
     */
    protected BaseAuroraDataBuffer(long length) {
        this(length, true);
    }

    protected BaseAuroraDataBuffer(long length, boolean initialize) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 0");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if (dataType() != DataType.UTF8)
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, dataType(), false);

        if (dataType() == DataType.DOUBLE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asDoublePointer();

            indexer = new DeviceDoubleIndexer((DoublePointer) pointer);

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.FLOAT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asFloatPointer();

            setIndexer(new DeviceFloatIndexer((FloatPointer) pointer));

            if (initialize)
                fillPointerWithZero();

        } else if (dataType() == DataType.HALF) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(new DeviceHalfIndexer((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BFLOAT16) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(new DeviceBfloat16Indexer((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.INT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();

            setIndexer(new DeviceIntIndexer((IntPointer) pointer));
            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.LONG) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();

            setIndexer(new DeviceLongIndexer((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(new DeviceByteIndexer((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.SHORT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(new DeviceShortIndexer((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UBYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(new DeviceUByteIndexer((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT16) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(new DeviceUShortIndexer((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT32) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();

            setIndexer(new DeviceUIntIndexer((IntPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT64) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();

            setIndexer(new DeviceLongIndexer((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BOOL) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBoolPointer();

            setIndexer(new DeviceBooleanIndexer((BooleanPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UTF8) {
            // we are allocating buffer as INT8 intentionally
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length(), INT8, false);
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length()).asBytePointer();

            setIndexer(new DeviceByteIndexer((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        }

        Nd4j.getDeallocatorService().pickObject(this);
    }

    public void actualizePointerAndIndexer() {
        val cptr = ptrDataBuffer.primaryBuffer();

        // skip update if pointers are equal
        if (cptr != null && pointer != null && cptr.address() == pointer.address())
            return;

        val t = dataType();
        if (t == DataType.BOOL) {
            pointer = new PagedPointer(cptr, length).asBoolPointer();
            setIndexer(new DeviceBooleanIndexer((BooleanPointer) pointer));
        } else if (t == DataType.UBYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(new DeviceUByteIndexer((BytePointer) pointer));
        } else if (t == DataType.BYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(new DeviceByteIndexer((BytePointer) pointer));
        } else if (t == DataType.UINT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(new DeviceUShortIndexer((ShortPointer) pointer));
        } else if (t == DataType.SHORT) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(new DeviceShortIndexer((ShortPointer) pointer));
        } else if (t == DataType.UINT32) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(new DeviceUIntIndexer((IntPointer) pointer));
        } else if (t == DataType.INT) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(new DeviceIntIndexer((IntPointer) pointer));
        } else if (t == DataType.UINT64) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(new DeviceLongIndexer((LongPointer) pointer));
        } else if (t == DataType.LONG) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(new DeviceLongIndexer((LongPointer) pointer));
        } else if (t == DataType.BFLOAT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(new DeviceBfloat16Indexer((ShortPointer) pointer));
        } else if (t == DataType.HALF) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(new DeviceHalfIndexer((ShortPointer) pointer));
        } else if (t == DataType.FLOAT) {
            pointer = new PagedPointer(cptr, length).asFloatPointer();
            setIndexer(new DeviceFloatIndexer((FloatPointer) pointer));
        } else if (t == DataType.DOUBLE) {
            pointer = new PagedPointer(cptr, length).asDoublePointer();
            setIndexer(new DeviceDoubleIndexer((DoublePointer) pointer));
        } else if (t == DataType.UTF8) {
            pointer = new PagedPointer(cptr, length()).asBytePointer();
            setIndexer(new DeviceByteIndexer((BytePointer) pointer));
        } else
            throw new IllegalArgumentException("Unknown datatype: " + dataType());
    }

    @Override
    public void setData(int[] data) {
        IntPointer temp = new IntPointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 4, 1, null);

    }

    @Override
    public void setData(float[] data) {
        FloatPointer temp = new FloatPointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 4, 1, null);

    }

    @Override
    public void setData(double[] data) {
        DoublePointer temp = new DoublePointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 8, 1, null);

    }

    @Override
    public void setData(long[] data) {
        LongPointer temp = new LongPointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 8, 1, null);

    }

    @Override
    public void setData(byte[] data) {
        super.setData(data);
    }

    @Override
    public void setData(short[] data) {
        super.setData(data);
    }

    @Override
    public void setData(boolean[] data) {
        super.setData(data);
    }

    @Override
    public Pointer addressPointer() {
        // we're fetching actual pointer right from C++
        val tempPtr = new PagedPointer(ptrDataBuffer.primaryBuffer());

        switch (this.type) {
            case DOUBLE: return tempPtr.asDoublePointer();
            case FLOAT: return tempPtr.asFloatPointer();
            case UINT16:
            case SHORT:
            case BFLOAT16:
            case HALF: return tempPtr.asShortPointer();
            case UINT32:
            case INT: return tempPtr.asIntPointer();
            case UBYTE:
            case BYTE: return tempPtr.asBytePointer();
            case UINT64:
            case LONG: return tempPtr.asLongPointer();
            case BOOL: return tempPtr.asBoolPointer();
            default: return tempPtr.asBytePointer();
        }
    }

    protected BaseAuroraDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();



        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if (dataType() == DataType.DOUBLE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asDoublePointer(); //new DoublePointer(length());
            indexer = new DeviceDoubleIndexer((DoublePointer) pointer);

        } else if (dataType() == DataType.FLOAT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asFloatPointer(); //new FloatPointer(length());
            setIndexer(new DeviceFloatIndexer((FloatPointer) pointer));

        } else if (dataType() == DataType.HALF) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new FloatPointer(length());
            setIndexer(new DeviceHalfIndexer((ShortPointer) pointer));

        } else if (dataType() == DataType.BFLOAT16) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new FloatPointer(length());
            setIndexer(new DeviceBfloat16Indexer((ShortPointer) pointer));
        } else if (dataType() == DataType.INT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(new DeviceIntIndexer((IntPointer) pointer));

        } else if (dataType() == DataType.UINT32) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(new DeviceUIntIndexer((IntPointer) pointer));

        } else if (dataType() == DataType.UINT64) {
            attached = true;
            parentWorkspace = workspace;

            // FIXME: need unsigned indexer here
            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new IntPointer(length());
            setIndexer(new DeviceLongIndexer((LongPointer) pointer));

        } else if (dataType() == DataType.LONG) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new LongPointer(length());
            setIndexer(new DeviceLongIndexer((LongPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBytePointer(); //new LongPointer(length());
            setIndexer(new DeviceByteIndexer((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBytePointer(); //new LongPointer(length());
            setIndexer(new DeviceUByteIndexer((BytePointer) pointer));
        } else if (dataType() == DataType.UINT16) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new IntPointer(length());
            setIndexer(new DeviceUShortIndexer((ShortPointer) pointer));

        } else if (dataType() == DataType.SHORT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new LongPointer(length());
            setIndexer(new DeviceShortIndexer((ShortPointer) pointer));
        } else if (dataType() == DataType.BOOL) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBoolPointer(); //new LongPointer(length());
            setIndexer(new DeviceBooleanIndexer((BooleanPointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new LongPointer(length());
            setIndexer(new DeviceLongIndexer((LongPointer) pointer));
        }

        // storing pointer into native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, dataType(), new PagedPointer(pointer, 0), null);

        // adding deallocator reference
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
    }

    public BaseAuroraDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);

        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, dataType(), new PagedPointer(pointer, 0), null);
        Nd4j.getDeallocatorService().pickObject(this);;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(float[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;

    }

    public BaseAuroraDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data, copy, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(float[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        FloatPointer temp = new FloatPointer(data);
        pointer = new FloatPointer(nativeOps.mallocDevice(data.length * 4, -1, 0));
        nativeOps.memcpySync(pointer, temp, data.length * 4, 1, null);

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.FLOAT, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        setIndexer(new DeviceFloatIndexer((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();

        length = data.length;
        underlyingLength = data.length;
    }

    public BaseAuroraDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asFloatPointer();
        FloatPointer temp = new FloatPointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 4, 1, null);

        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, dataType(), new PagedPointer(pointer, 0), null);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        setIndexer(new DeviceFloatIndexer((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseAuroraDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asDoublePointer();
        DoublePointer temp = new DoublePointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 8, 1, null);

        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, dataType(), new PagedPointer(pointer, 0), null);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        indexer = new DeviceDoubleIndexer((DoublePointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    public BaseAuroraDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asIntPointer();
        IntPointer temp = new IntPointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 4, 1, null);

        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, dataType(), new PagedPointer(pointer, 0), null);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        indexer = new DeviceIntIndexer((IntPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseAuroraDataBuffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asLongPointer();
        LongPointer temp = new LongPointer(data);
        nativeOps.memcpySync(pointer, temp, data.length * 8, 1, null);

        ptrDataBuffer = OpaqueDataBuffer.externalizedDataBuffer(length, dataType(), new PagedPointer(pointer, 0), null);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        indexer = new DeviceLongIndexer((LongPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(double[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    public BaseAuroraDataBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data, copy, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(double[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        DoublePointer temp = new DoublePointer(data);
        pointer = new DoublePointer(nativeOps.mallocDevice(data.length * 8, -1, 0));
        nativeOps.memcpySync(pointer, temp, data.length * 8, 1, null);
        indexer = new DeviceDoubleIndexer((DoublePointer) pointer);

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.DOUBLE, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(int[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(int[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        IntPointer temp = new IntPointer(data);
        pointer = new IntPointer(nativeOps.mallocDevice(data.length * 4, -1, 0));
        nativeOps.memcpySync(pointer, temp, data.length * 4, 1, null);
        setIndexer(new DeviceIntIndexer((IntPointer) pointer));

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.INT32, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseAuroraDataBuffer(long[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        LongPointer temp = new LongPointer(data);
        pointer = new LongPointer(nativeOps.mallocDevice(data.length * 8, -1, 0));
        nativeOps.memcpySync(pointer, temp, data.length * 8, 1, null);
        setIndexer(new DeviceLongIndexer((LongPointer) pointer));

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.INT64, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     */
    public BaseAuroraDataBuffer(double[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseAuroraDataBuffer(int[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseAuroraDataBuffer(float[] data) {
        this(data, true);
    }

    public BaseAuroraDataBuffer(float[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }

    @Override
    protected void release() {
        ptrDataBuffer.closeBuffer();
        super.release();
    }

    /**
     * Reallocate the native memory of the buffer
     * @param length the new length of the buffer
     * @return this databuffer
     * */
    @Override
    public DataBuffer reallocate(long length) {
        val oldPointer = ptrDataBuffer.primaryBuffer();

        if (isAttached()) {
            val capacity = length * getElementSize();
            val nPtr = getParentWorkspace().alloc(capacity, dataType(), false);
            this.ptrDataBuffer.setPrimaryBuffer(new PagedPointer(nPtr, 0), length);

            switch (dataType()) {
                case BOOL:
                    pointer = nPtr.asBoolPointer();
                    indexer = new DeviceBooleanIndexer((BooleanPointer) pointer);
                    break;
                case UTF8:
                case BYTE:
                case UBYTE:
                    pointer = nPtr.asBytePointer();
                    indexer = new DeviceByteIndexer((BytePointer) pointer);
                    break;
                case UINT16:
                case SHORT:
                    pointer = nPtr.asShortPointer();
                    indexer = new DeviceShortIndexer((ShortPointer) pointer);
                    break;
                case UINT32:
                    pointer = nPtr.asIntPointer();
                    indexer = new DeviceUIntIndexer((IntPointer) pointer);
                    break;
                case INT:
                    pointer = nPtr.asIntPointer();
                    indexer = new DeviceIntIndexer((IntPointer) pointer);
                    break;
                case DOUBLE:
                    pointer = nPtr.asDoublePointer();
                    indexer = new DeviceDoubleIndexer((DoublePointer) pointer);
                    break;
                case FLOAT:
                    pointer = nPtr.asFloatPointer();
                    indexer = new DeviceFloatIndexer((FloatPointer) pointer);
                    break;
                case HALF:
                    pointer = nPtr.asShortPointer();
                    indexer = new DeviceHalfIndexer((ShortPointer) pointer);
                    break;
                case BFLOAT16:
                    pointer = nPtr.asShortPointer();
                    indexer = new DeviceBfloat16Indexer((ShortPointer) pointer);
                    break;
                case UINT64:
                case LONG:
                    pointer = nPtr.asLongPointer();
                    indexer = new DeviceLongIndexer((LongPointer) pointer);
                    break;
            }

            nativeOps.memcpySync(pointer, oldPointer, this.length() * getElementSize(), 3, null);
            workspaceGenerationId = getParentWorkspace().getGenerationId();
        } else {
            this.ptrDataBuffer.expand(length);
            val nPtr = new PagedPointer(this.ptrDataBuffer.primaryBuffer(), length);

            switch (dataType()) {
                case BOOL:
                    pointer = nPtr.asBoolPointer();
                    indexer = new DeviceBooleanIndexer((BooleanPointer) pointer);
                    break;
                case UTF8:
                case BYTE:
                case UBYTE:
                    pointer = nPtr.asBytePointer();
                    indexer = new DeviceByteIndexer((BytePointer) pointer);
                    break;
                case UINT16:
                case SHORT:
                    pointer = nPtr.asShortPointer();
                    indexer = new DeviceShortIndexer((ShortPointer) pointer);
                    break;
                case UINT32:
                    pointer = nPtr.asIntPointer();
                    indexer = new DeviceUIntIndexer((IntPointer) pointer);
                    break;
                case INT:
                    pointer = nPtr.asIntPointer();
                    indexer = new DeviceIntIndexer((IntPointer) pointer);
                    break;
                case DOUBLE:
                    pointer = nPtr.asDoublePointer();
                    indexer = new DeviceDoubleIndexer((DoublePointer) pointer);
                    break;
                case FLOAT:
                    pointer = nPtr.asFloatPointer();
                    indexer = new DeviceFloatIndexer((FloatPointer) pointer);
                    break;
                case HALF:
                    pointer = nPtr.asShortPointer();
                    indexer = new DeviceHalfIndexer((ShortPointer) pointer);
                    break;
                case BFLOAT16:
                    pointer = nPtr.asShortPointer();
                    indexer = new DeviceBfloat16Indexer((ShortPointer) pointer);
                    break;
                case UINT64:
                case LONG:
                    pointer = nPtr.asLongPointer();
                    indexer = new DeviceLongIndexer((LongPointer) pointer);
                    break;
            }
        }

        this.underlyingLength = length;
        this.length = length;
        return this;
    }

    @Override
    public void syncToPrimary(){
        ptrDataBuffer.syncToPrimary();
    }

    @Override
    public void syncToSpecial(){
        ptrDataBuffer.syncToSpecial();
    }
}
