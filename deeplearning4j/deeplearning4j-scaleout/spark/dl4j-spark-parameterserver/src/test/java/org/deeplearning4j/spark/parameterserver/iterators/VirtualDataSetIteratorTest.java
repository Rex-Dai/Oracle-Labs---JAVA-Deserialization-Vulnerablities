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

package org.deeplearning4j.spark.parameterserver.iterators;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.resources.Downloader;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
@Slf4j
public class VirtualDataSetIteratorTest {
    @BeforeEach
    public void setUp() throws Exception {}



    @BeforeAll
    @SneakyThrows
    public static void beforeAll() {
        if(Platform.isWindows()) {
            File hadoopHome = new File(System.getProperty("java.io.tmpdir"),"hadoop-tmp");
            File binDir = new File(hadoopHome,"bin");
            if(!binDir.exists())
                binDir.mkdirs();
            File outputFile = new File(binDir,"winutils.exe");
            if(!outputFile.exists()) {
                log.info("Fixing spark for windows");
                Downloader.download("winutils.exe",
                        URI.create("https://github.com/cdarlint/winutils/blob/master/hadoop-2.6.5/bin/winutils.exe?raw=true").toURL(),
                        outputFile,"db24b404d2331a1bec7443336a5171f1",3);
            }

            System.setProperty("hadoop.home.dir", hadoopHome.getAbsolutePath());
        }
    }

    @Test
    public void testSimple1() throws Exception {
        List<Iterator<DataSet>> iterators = new ArrayList<>();

        List<DataSet> first = new ArrayList<>();
        List<DataSet> second = new ArrayList<>();

        for (int i = 0; i < 100; i++) {
            INDArray features = Nd4j.create(100).assign(i);
            INDArray labels = Nd4j.create(10).assign(i);
            DataSet ds = new DataSet(features, labels);

            if (i < 25)
                first.add(ds);
            else
                second.add(ds);
        }

        iterators.add(first.iterator());
        iterators.add(second.iterator());

        VirtualDataSetIterator vdsi = new VirtualDataSetIterator(iterators);
        int cnt = 0;
        while (vdsi.hasNext()) {
            DataSet ds = vdsi.next();

            assertEquals((double) cnt, ds.getFeatures().meanNumber().doubleValue(), 0.0001);
            assertEquals((double) cnt, ds.getLabels().meanNumber().doubleValue(), 0.0001);

            cnt++;
        }

        assertEquals(100, cnt);
    }
}
