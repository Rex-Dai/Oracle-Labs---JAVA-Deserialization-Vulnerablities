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

package org.nd4j.common.primitives;

import java.io.Serializable;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class SynchronizedObject<T> implements Serializable {
    protected T value;
    protected transient ReentrantReadWriteLock lock;

    public SynchronizedObject() {
        lock = new ReentrantReadWriteLock();
    }

    public SynchronizedObject(T value) {
        this();

        this.set(value);
    }

    /**
     * This method returns stored value via read lock
     * @return
     */
    public final T get() {
        try {
            lock.readLock().lock();

            return value;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * This method updates stored value via write lock
     * @param value
     */
    public final void set(T value) {
        try {
            lock.writeLock().lock();

            this.value = value;
        } finally {
            lock.writeLock().unlock();
        }
    }
}
