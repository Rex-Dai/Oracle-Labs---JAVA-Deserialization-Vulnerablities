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

package org.nd4j.aeron.ipc;

import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.agrona.concurrent.SigIntBarrier;

import static java.lang.System.setProperty;
import static org.agrona.concurrent.UnsafeBuffer.DISABLE_BOUNDS_CHECKS_PROP_NAME;
import static org.nd4j.aeron.ipc.AeronUtil.setSystemPropertyIfNotSet;

public class LowLatencyMediaDriver {

    private LowLatencyMediaDriver() {}

    @SuppressWarnings("checkstyle:UncommentedMain")
    public static void main(final String... args) {
        MediaDriver.main(args);
        setSystemPropertyIfNotSet(DISABLE_BOUNDS_CHECKS_PROP_NAME, "true");
        setSystemPropertyIfNotSet("aeron.mtu.length", "16384");
        setSystemPropertyIfNotSet("aeron.socket.so_sndbuf", "2097152");
        setSystemPropertyIfNotSet("aeron.socket.so_rcvbuf", "2097152");
        setSystemPropertyIfNotSet("aeron.rcv.initial.window.length", "2097152");

        final MediaDriver.Context ctx =
                new MediaDriver.Context().threadingMode(ThreadingMode.DEDICATED).dirDeleteOnStart(true)
                        .dirDeleteOnShutdown(true)
                        .termBufferSparseFile(false).conductorIdleStrategy(new BusySpinIdleStrategy())
                        .receiverIdleStrategy(new BusySpinIdleStrategy())
                        .senderIdleStrategy(new BusySpinIdleStrategy());

        try (MediaDriver ignored = MediaDriver.launch(ctx)) {
            new SigIntBarrier().await();

        }
    }

}
