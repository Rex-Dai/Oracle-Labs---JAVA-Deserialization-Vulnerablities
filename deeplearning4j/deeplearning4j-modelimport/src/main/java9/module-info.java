open module deeplearning4j.modelimport {
    requires commons.io;
    requires gson;
    requires guava;
    requires org.apache.commons.lang3;
    requires org.bytedeco.javacpp;
    requires protobuf;
    requires resources;
    requires slf4j.api;
    requires deeplearning4j.nn;
    requires jackson;
    requires nd4j.api;
    requires nd4j.common;
    requires org.bytedeco.hdf5;
    exports org.deeplearning4j.frameworkimport.keras.keras;
    exports org.deeplearning4j.frameworkimport.keras.keras.config;
    exports org.deeplearning4j.frameworkimport.keras.keras.exceptions;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.advanced.activations;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.convolutional;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.core;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.custom;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.embeddings;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.local;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.noise;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.normalization;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.pooling;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.recurrent;
    exports org.deeplearning4j.frameworkimport.keras.keras.layers.wrappers;
    exports org.deeplearning4j.frameworkimport.keras.keras.preprocessing.sequence;
    exports org.deeplearning4j.frameworkimport.keras.keras.preprocessing.text;
    exports org.deeplearning4j.frameworkimport.keras.keras.preprocessors;
    exports org.deeplearning4j.frameworkimport.keras.keras.utils;
}
