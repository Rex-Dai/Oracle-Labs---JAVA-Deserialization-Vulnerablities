package model

import org.apache.commons.lang3.tuple.ImmutablePair
import org.apache.commons.lang3.tuple.Pair
import java.io.File
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.learning.config.AdaGrad
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import javax.swing._
import java.awt._
import java.awt.image.BufferedImage
import java.util._
import java.util
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator

import scala.collection.JavaConversions._

class AutoEncoder {
  val dataPath = new File("./dataSet.csv")

  val batchSize = 10
  val numLabelClasses = 54

  // training data
  val trainRR = new CSVSequenceRecordReader(0, ", ")
  trainRR.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 0, 87))
  val trainIter = new SequenceRecordReaderDataSetIterator(trainRR, batchSize, numLabelClasses, 1)


  val conf = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .weightInit(WeightInit.XAVIER)
    .updater(new AdaGrad(0.05))
    .activation(Activation.RELU)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .l2(0.0001)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(54).nOut(27)
      .build())
    .layer(1, new DenseLayer.Builder().nIn(27).nOut(10)
      .build())
    .layer(2, new DenseLayer.Builder().nIn(10).nOut(27)
      .build())
    .layer(3, new OutputLayer.Builder().nIn(27).nOut(54)
      .lossFunction(LossFunctions.LossFunction.MSE)
      .build())
    .build()

  val net = new MultiLayerNetwork(conf)
  net.setListeners(new ScoreIterationListener(1))


  val featuresTrain = new util.ArrayList[INDArray]
  val featuresTest = new util.ArrayList[INDArray]
  val labelsTest = new util.ArrayList[INDArray]


  while(tranIter.hasNext()){
    val next = trainIter.next()
    val split = next.splitTestAndTrain(8, 2)  //8/2 split (from miniBatch = 10)
    featuresTrain.add(split.getTrain().getFeatures())
    val dsTest = split.getTest()
    featuresTest.add(dsTest.getFeatures())
    val indexes = Nd4j.argMax(dsTest.getLabels(),1) //Convert from one-hot representation -> index
    labelsTest.add(indexes)
  }

  def main(args: Array[String]): Unit = {
    val nEpochs = 50
    (1 to nEpochs).foreach{ epoch =>
      featuresTrain.forEach( data => net.fit(data, data))
      println("Epoch " + epoch + " complete");
    }
  }

}
