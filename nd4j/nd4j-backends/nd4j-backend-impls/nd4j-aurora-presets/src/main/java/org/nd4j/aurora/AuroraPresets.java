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

package org.nd4j.aurora;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.NoException;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;

import java.util.List;

/**
 *
 * @author saudet
 */
@Properties(
        value = {
                @Platform(
                        value = "linux-x86_64",
                        cinclude = "ve_offload.h",
                        link = "veo@.0",
                        includepath = "/opt/nec/ve/veos/include/",
                        linkpath = "/opt/nec/ve/veos/lib64/",
                        library = "aurora",
                        resource = {"aurora", "libaurora.so","nd4jaurora"}
                )
        },
        target = "org.nd4j.aurora.Aurora"
)
@NoException
public class AuroraPresets implements InfoMapper, BuildEnabled, LoadEnabled {

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("char").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]", "String"));
    }

    @Override
    public void init(ClassProperties properties) {
        if(!Loader.isLoadLibraries()) {
            return;
        }


    }
}