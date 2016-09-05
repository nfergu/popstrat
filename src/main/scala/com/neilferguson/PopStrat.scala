package com.neilferguson

import hex.FrameSplitter
import hex.deeplearning.DeepLearning
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.h2o.H2OContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{Genotype, GenotypeAllele}
import water.Key

import scala.collection.JavaConverters._
import scala.collection.immutable.Range.inclusive
import scala.io.Source

import org.apache.spark.sql.types.DataTypes
import hex._
import water.fvec._
import water.support._
import _root_.hex.Distribution.Family
import _root_.hex.deeplearning.DeepLearningModel
import _root_.hex.tree.gbm.GBMModel
import _root_.hex.{Model, ModelMetricsBinomial}
object PopStrat {

  def main(args: Array[String]): Unit = {

    val genotypeFile = args(0)
    val panelFile = args(1)

    val master = if (args.length > 2) Some(args(2)) else None
    val conf = new SparkConf().setAppName("PopStrat")
    master.foreach(conf.setMaster)
    val sc = new SparkContext(conf)

    // Create a set of the populations that we want to predict
    // Then create a map of sample ID -> population so that we can filter out the samples we're not interested in
    val populations = Set("GBR", "ASW", "CHB")
    def extract(file: String, filter: (String, String) => Boolean): Map[String,String] = {
      Source.fromFile(file).getLines().map(line => {
        val tokens = line.split("\t").toList
        tokens(0) -> tokens(1)
      }).toMap.filter(tuple => filter(tuple._1, tuple._2))
    }
    val panel: Map[String,String] = extract(panelFile, (sampleID: String, pop: String) => populations.contains(pop))

    // Load the ADAM genotypes from the parquet file(s)
    // Next, filter the genotypes so that we're left with only those in the populations we're interested in
    val allGenotypes: RDD[Genotype] = sc.loadGenotypes(genotypeFile)
    val genotypes: RDD[Genotype] = allGenotypes.filter(genotype => {panel.contains(genotype.getSampleId)})

    // Convert the Genotype objects to our own SampleVariant objects to try and conserve memory
    case class SampleVariant(sampleId: String, variantId: Int, alternateCount: Int)
    def variantId(genotype: Genotype): String = {
      val name = genotype.getVariant.getContig.getContigName
      val start = genotype.getVariant.getStart
      val end = genotype.getVariant.getEnd
      s"$name:$start:$end"
    }
    def alternateCount(genotype: Genotype): Int = {
      genotype.getAlleles.asScala.count(_ != GenotypeAllele.Ref)
    }
    def toVariant(genotype: Genotype): SampleVariant = {
      // Intern sample IDs as they will be repeated a lot
      new SampleVariant(genotype.getSampleId.intern(), variantId(genotype).hashCode(), alternateCount(genotype))
    }
    val variantsRDD: RDD[SampleVariant] = genotypes.map(toVariant)

    // Group the variants by sample ID so we can process the variants sample-by-sample
    // Then get the total number of samples. This will be used to find variants that are missing for some samples.
    // Group the variants by variant ID and filter out those variants that are missing from some samples
    val variantsBySampleId: RDD[(String, Iterable[SampleVariant])] = variantsRDD.groupBy(_.sampleId)
    val sampleCount: Long = variantsBySampleId.count()
    println("Found " + sampleCount + " samples")
    val variantsByVariantId: RDD[(Int, Iterable[SampleVariant])] = variantsRDD.groupBy(_.variantId).filter {
      case (_, sampleVariants) => sampleVariants.size == sampleCount
    }

    // Make a map of variant ID -> count of samples with an alternate count of greater than zero
    // then filter out those variants that are not in our desired frequency range. The objective here is simply to
    // reduce the number of dimensions in the data set to make it easier to train the model.
    // The specified range is fairly arbitrary and was chosen based on the fact that it includes a reasonable
    // number of variants, but not too many.
    val variantFrequencies: collection.Map[Int, Int] = variantsByVariantId.map {
      case (variantId, sampleVariants) => (variantId, sampleVariants.count(_.alternateCount > 0))
    }.collectAsMap()
    val permittedRange = inclusive(11, 11)
    val filteredVariantsBySampleId: RDD[(String, Iterable[SampleVariant])] = variantsBySampleId.map {
      case (sampleId, sampleVariants) =>
        val filteredSampleVariants = sampleVariants.filter(variant => permittedRange.contains(
          variantFrequencies.getOrElse(variant.variantId, -1)))
        (sampleId, filteredSampleVariants)
    }

    // Sort the variants for each sample ID. Each sample should now have the same number of sorted variants.
    // All items in the RDD should now have the same variants in the same order so we can just use the first
    // one to construct our header
    // Next construct the rows of our SchemaRDD from the variants
    val sortedVariantsBySampleId: RDD[(String, Array[SampleVariant])] = filteredVariantsBySampleId.map {
      case (sampleId, variants) =>
        (sampleId, variants.toArray.sortBy(_.variantId))
    }
    val header = DataTypes.createStructType(Array(DataTypes.createStructField("Region", DataTypes.StringType,false)) ++
      sortedVariantsBySampleId.first()._2.map(variant => {DataTypes.createStructField(variant.variantId.toString,DataTypes.IntegerType,false)}))
    val rowRDD: RDD[Row] = sortedVariantsBySampleId.map {
      case (sampleId, sortedVariants) =>
        val region: Array[String] = Array(panel.getOrElse(sampleId, "Unknown"))
        val alternateCounts: Array[Int] = sortedVariants.map(_.alternateCount)
        Row.fromSeq(region ++ alternateCounts)
    }

    // Create the SchemaRDD from the header and rows and convert the SchemaRDD into a H2O dataframe
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //val dataFrame=sqlContext.createDataFrame(rowRDD, header)
    val schemaRDD = sqlContext.applySchema(rowRDD, header)
    val h2oContext = new H2OContext(sc).start()
    import h2oContext._ 
    val dataFrame1 =h2oContext.asH2OFrame(schemaRDD)
    val dataFrame=H2OFrameSupport.allStringVecToCategorical(dataFrame1)

    // Split the dataframe into 50% training, 30% test, and 20% validation data
    val frameSplitter = new FrameSplitter(dataFrame, Array(.5, .3), Array("training", "test", "validation").map(Key.make[Frame](_)), null)
    water.H2O.submitTask(frameSplitter)
    val splits = frameSplitter.getResult
    val training = splits(0)
    val validation = splits(2)

    // Set the parameters for our deep learning model.
    val deepLearningParameters = new DeepLearningParameters()
    deepLearningParameters._train = training._key
    deepLearningParameters._valid = validation._key
    deepLearningParameters._response_column = "Region"
    deepLearningParameters._epochs = 10
    deepLearningParameters._activation = Activation.RectifierWithDropout
    deepLearningParameters._hidden = Array[Int](100,100)

    // Train the deep learning model
    val deepLearning = new DeepLearning(deepLearningParameters)
    val deepLearningModel = deepLearning.trainModel.get

    // Score the model against the entire dataset (training, test, and validation data)
    // This causes the confusion matrix to be printed
    deepLearningModel.score(dataFrame)

  }

}
