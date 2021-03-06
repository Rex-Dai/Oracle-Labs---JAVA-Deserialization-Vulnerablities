open module nd4j.parameter.server.node {
    requires commons.io;
    requires commons.math3;
    requires commons.net;
    requires guava;
    requires nd4j.aeron;
    requires org.apache.commons.lang3;
    requires slf4j.api;
    requires io.aeron.all;
    requires io.reactivex.rxjava2;
    requires nd4j.api;
    requires nd4j.common;
    requires nd4j.parameter.server;
    requires org.reactivestreams;
    exports org.nd4j.parameterserver.distributed;
    exports org.nd4j.parameterserver.distributed.conf;
    exports org.nd4j.parameterserver.distributed.enums;
    exports org.nd4j.parameterserver.distributed.logic;
    exports org.nd4j.parameterserver.distributed.logic.completion;
    exports org.nd4j.parameterserver.distributed.logic.retransmission;
    exports org.nd4j.parameterserver.distributed.logic.routing;
    exports org.nd4j.parameterserver.distributed.logic.sequence;
    exports org.nd4j.parameterserver.distributed.logic.storage;
    exports org.nd4j.parameterserver.distributed.messages;
    exports org.nd4j.parameterserver.distributed.messages.aggregations;
    exports org.nd4j.parameterserver.distributed.messages.complete;
    exports org.nd4j.parameterserver.distributed.messages.intercom;
    exports org.nd4j.parameterserver.distributed.messages.requests;
    exports org.nd4j.parameterserver.distributed.training;
    exports org.nd4j.parameterserver.distributed.training.chains;
    exports org.nd4j.parameterserver.distributed.training.impl;
    exports org.nd4j.parameterserver.distributed.transport;
    exports org.nd4j.parameterserver.distributed.util;
    exports org.nd4j.parameterserver.distributed.v2;
    exports org.nd4j.parameterserver.distributed.v2.chunks;
    exports org.nd4j.parameterserver.distributed.v2.chunks.impl;
    exports org.nd4j.parameterserver.distributed.v2.enums;
    exports org.nd4j.parameterserver.distributed.v2.messages;
    exports org.nd4j.parameterserver.distributed.v2.messages.history;
    exports org.nd4j.parameterserver.distributed.v2.messages.impl;
    exports org.nd4j.parameterserver.distributed.v2.messages.impl.base;
    exports org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake;
    exports org.nd4j.parameterserver.distributed.v2.messages.pairs.params;
    exports org.nd4j.parameterserver.distributed.v2.messages.pairs.ping;
    exports org.nd4j.parameterserver.distributed.v2.transport;
    exports org.nd4j.parameterserver.distributed.v2.transport.impl;
    exports org.nd4j.parameterserver.distributed.v2.util;
    exports org.nd4j.parameterserver.node;
    provides org.nd4j.parameterserver.distributed.training.TrainingDriver with org.nd4j.parameterserver.distributed.training.impl.SkipGramTrainer, org.nd4j.parameterserver.distributed.training.impl.CbowTrainer;
}
