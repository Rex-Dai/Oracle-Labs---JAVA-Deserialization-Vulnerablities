open module deeplearning4j.nn {
    requires byteunits;
    requires commons.io;
    requires deeplearning4j.utility.iterators;
    requires java.management;
    requires nd4j.cpu.api;
    requires org.apache.commons.lang3;
    requires oshi.core;
    requires resources;
    requires commons.math3;
    requires fastutil;
    requires guava;
    requires jackson;
    requires nd4j.api;
    requires nd4j.common;
    requires org.bytedeco.javacpp;
    requires slf4j.api;
    exports org.deeplearning4j.earlystopping;
    exports org.deeplearning4j.earlystopping.listener;
    exports org.deeplearning4j.earlystopping.saver;
    exports org.deeplearning4j.earlystopping.scorecalc;
    exports org.deeplearning4j.earlystopping.scorecalc.base;
    exports org.deeplearning4j.earlystopping.termination;
    exports org.deeplearning4j.earlystopping.trainer;
    exports org.deeplearning4j.eval;
    exports org.deeplearning4j.eval.curves;
    exports org.deeplearning4j.eval.meta;
    exports org.deeplearning4j.exception;
    exports org.deeplearning4j.gradientcheck;
    exports org.deeplearning4j.nn.adapters;
    exports org.deeplearning4j.nn.api;
    exports org.deeplearning4j.nn.api.layers;
    exports org.deeplearning4j.nn.conf;
    exports org.deeplearning4j.nn.conf.constraint;
    exports org.deeplearning4j.nn.conf.distribution;
    exports org.deeplearning4j.nn.conf.distribution.serde;
    exports org.deeplearning4j.nn.conf.dropout;
    exports org.deeplearning4j.nn.conf.graph;
    exports org.deeplearning4j.nn.conf.graph.rnn;
    exports org.deeplearning4j.nn.conf.inputs;
    exports org.deeplearning4j.nn.conf.layers;
    exports org.deeplearning4j.nn.conf.layers.convolutional;
    exports org.deeplearning4j.nn.conf.layers.misc;
    exports org.deeplearning4j.nn.conf.layers.objdetect;
    exports org.deeplearning4j.nn.conf.layers.recurrent;
    exports org.deeplearning4j.nn.conf.layers.samediff;
    exports org.deeplearning4j.nn.conf.layers.util;
    exports org.deeplearning4j.nn.conf.layers.variational;
    exports org.deeplearning4j.nn.conf.layers.wrapper;
    exports org.deeplearning4j.nn.conf.memory;
    exports org.deeplearning4j.nn.conf.misc;
    exports org.deeplearning4j.nn.conf.module;
    exports org.deeplearning4j.nn.conf.ocnn;
    exports org.deeplearning4j.nn.conf.preprocessor;
    exports org.deeplearning4j.nn.conf.serde;
    exports org.deeplearning4j.nn.conf.serde.format;
    exports org.deeplearning4j.nn.conf.serde.legacy;
    exports org.deeplearning4j.nn.conf.stepfunctions;
    exports org.deeplearning4j.nn.conf.weightnoise;
    exports org.deeplearning4j.nn.gradient;
    exports org.deeplearning4j.nn.graph;
    exports org.deeplearning4j.nn.graph.util;
    exports org.deeplearning4j.nn.graph.vertex;
    exports org.deeplearning4j.nn.graph.vertex.impl;
    exports org.deeplearning4j.nn.graph.vertex.impl.rnn;
    exports org.deeplearning4j.nn.layers;
    exports org.deeplearning4j.nn.layers.convolution;
    exports org.deeplearning4j.nn.layers.convolution.subsampling;
    exports org.deeplearning4j.nn.layers.convolution.upsampling;
    exports org.deeplearning4j.nn.layers.feedforward;
    exports org.deeplearning4j.nn.layers.feedforward.autoencoder;
    exports org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive;
    exports org.deeplearning4j.nn.layers.feedforward.dense;
    exports org.deeplearning4j.nn.layers.feedforward.elementwise;
    exports org.deeplearning4j.nn.layers.feedforward.embedding;
    exports org.deeplearning4j.nn.layers.mkldnn;
    exports org.deeplearning4j.nn.layers.normalization;
    exports org.deeplearning4j.nn.layers.objdetect;
    exports org.deeplearning4j.nn.layers.ocnn;
    exports org.deeplearning4j.nn.layers.pooling;
    exports org.deeplearning4j.nn.layers.recurrent;
    exports org.deeplearning4j.nn.layers.samediff;
    exports org.deeplearning4j.nn.layers.training;
    exports org.deeplearning4j.nn.layers.util;
    exports org.deeplearning4j.nn.layers.variational;
    exports org.deeplearning4j.nn.layers.wrapper;
    exports org.deeplearning4j.nn.multilayer;
    exports org.deeplearning4j.nn.params;
    exports org.deeplearning4j.nn.transferlearning;
    exports org.deeplearning4j.nn.updater;
    exports org.deeplearning4j.nn.updater.graph;
    exports org.deeplearning4j.nn.weights;
    exports org.deeplearning4j.nn.weights.embeddings;
    exports org.deeplearning4j.nn.workspace;
    exports org.deeplearning4j.optimize;
    exports org.deeplearning4j.optimize.api;
    exports org.deeplearning4j.optimize.listeners;
    exports org.deeplearning4j.optimize.listeners.callbacks;
    exports org.deeplearning4j.optimize.solvers;
    exports org.deeplearning4j.optimize.solvers.accumulation;
    exports org.deeplearning4j.optimize.solvers.accumulation.encoding;
    exports org.deeplearning4j.optimize.solvers.accumulation.encoding.residual;
    exports org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold;
    exports org.deeplearning4j.optimize.stepfunctions;
    exports org.deeplearning4j.preprocessors;
    exports org.deeplearning4j.util;
}
