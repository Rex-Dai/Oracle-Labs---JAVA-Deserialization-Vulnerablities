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

package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Queue;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

@Slf4j
public class BasicGradientsAccumulator implements GradientsAccumulator {

    protected MessageHandler handler;

    // here we'll store messages coming from "somewhere else"
    protected transient IndexedTail gradients;

    // this field stores current accumulated
    protected transient INDArray storage;

    // this field stores updates ready-to-apply
    protected transient INDArray updates;

    // this counter tracks number of messages generated by this accumulation
    protected transient AtomicLong ownCounter = new AtomicLong(0);

    // this counter tracks number of messages received from somewhere
    protected transient AtomicLong extCounter = new AtomicLong(0);

    // FIXME: this mechanics should be improved i think.
    protected long[] shape;
    protected char ordering;

    protected int parties = 0;
    protected CyclicBarrier barrier;
    protected AtomicLong firstOne = new AtomicLong(-1L);
    protected List<INDArray> candidates = new CopyOnWriteArrayList<>();

    protected ReentrantReadWriteLock updatesLock = new ReentrantReadWriteLock();
    protected AtomicBoolean hasSomething = new AtomicBoolean(false);

    /**
     * Creates new GradientsAccumulator with starting threshold of 1e-3
     */
    public BasicGradientsAccumulator(int parties) {
        this(parties, new LocalHandler());
    }

    /**
     * Creates new GradientsAccumulator with custom starting threshold
     *
     * @param handler MessageHandler instance that'll be used for communication purposes
     */
    public BasicGradientsAccumulator(int parties, @NonNull MessageHandler handler) {
        this.gradients = new IndexedTail(parties);
        this.handler = handler;

        this.handler.initialize(this);
        this.parties = parties;
        barrier = new CyclicBarrier(parties);
    }

    @Override
    public IndexedTail getExternalSource() {
        return gradients;
    }

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     */
    @Override
    public void applyUpdate(StepFunction function, INDArray params, INDArray grad, boolean isFinalStep) {

        try {
            updatesLock.readLock().lock();

            firstOne.compareAndSet(-1L, Thread.currentThread().getId());

            if (hasSomething.get())
                function.step(params, updates);

            barrier.await();
            if (firstOne.get() == Thread.currentThread().getId()) {
                // one thread just nullifies this array
                updates.assign(0.0);
                hasSomething.set(false);

                firstOne.set(-1L);
            }

            updatesLock.readLock().unlock();
            barrier.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (BrokenBarrierException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void markExternalUpdates(boolean updatesAvailable) {
        // no-op
    }

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     */
    @Override
    public void applyUpdate(StepFunction function, INDArray params, INDArray grad, double alpha) {

        try {
            updatesLock.readLock().lock();

            firstOne.compareAndSet(-1L, Thread.currentThread().getId());

            if (hasSomething.get())
                function.step(params, updates, alpha);

            barrier.await();
            if (firstOne.get() == Thread.currentThread().getId()) {
                // one thread just nullifies this array
                updates.assign(0.0);
                hasSomething.set(false);

                firstOne.set(-1L);
            }

            updatesLock.readLock().unlock();
            barrier.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (BrokenBarrierException e) {
            throw new RuntimeException(e);
	}
    }

    /**
     * This method accepts updates suitable for StepFunction, and accumulates/propagates it across all workers
     *
     * @param array
     */
    @Override
    public void storeUpdate(INDArray array, int iterationNumber, int epochNumber) {
        /*
            Here we want to do 4 things:
            1) update accumulated values
            2) invoke extractor, that'll (optionally) pull all updates above threshold
            3) ???
            4) PROFIT!
         */

        try {
            // commit should happen in each individual thread
            Nd4j.getExecutioner().commit();
            firstOne.compareAndSet(-1L, Thread.currentThread().getId());

            // TODO: since we know number of elements in advance, we don't really need CopyOnWrite list here.
            candidates.add(array);
            barrier.await();

            if (firstOne.get() == Thread.currentThread().getId()) {
                // if accum is null, let's just create it here
                if (storage == null) {
                    // we don't want state array to be attached to any workspace
                    shape = array.shape();
                    ordering = array.ordering();

                    try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        // TODO: if p2p isn't supported, this should become HOST-only allocation
                        storage = Nd4j.create(shape, ordering);
                    }
                }


                // accumulate our values, a
                //storage.addi(array);
                Nd4j.accumulate(storage, candidates);

                // we ensure storage was updated successfully
                Nd4j.getExecutioner().commit();

                // if there's something to send - send it. Skip otherwise!!!
                if (handler.broadcastUpdates(storage, iterationNumber, epochNumber)) {
                    ownCounter.getAndIncrement();
                }

                // reset "first one" :)
                firstOne.set(-1L);
                candidates.clear();
            }

            barrier.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (BrokenBarrierException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * This method accepts updates suitable for StepFunction and puts them to the queue, which is used in backpropagation loop
     *
     * PLEASE NOTE: array is expected to be ready for use and match params dimensionality
     *
     * @param array
     */
    @Override
    public void receiveUpdate(INDArray array) {
        extCounter.getAndIncrement();

        updatesLock.writeLock().lock();

        if (updates == null) {
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                // TODO: this one has to be HOST-only if P2P is NOT supported
                updates = Nd4j.create(array.shape(), array.ordering());
            }
        }

        hasSomething.compareAndSet(false, true);

        // if P2P is NOT supported - this call should be handled with cpu
        updates.addi(array);

        // we have to ensure, all operations were finished here
        Nd4j.getExecutioner().commit();

        updatesLock.writeLock().unlock();
    }


    /**
     * This method resets all accumulated updates (if any)
     */
    @Override
    public void reset() {
        updatesLock.writeLock().lock();

        if (storage != null) {
            storage.assign(0.0f);
        }

        if (updates != null)
            updates.assign(0.0f);

        updatesLock.writeLock().unlock();
    }

    /**
     * This method does initialization of given worker wrt Thread-Device Affinity
     */
    @Override
    public void touch() {
        // no-op
    }

    @Override
    public void setExternalSource(IndexedTail source) {
        gradients = source;
    }


    @Override
    public boolean hasAnything() {
        return false;
    }
}
