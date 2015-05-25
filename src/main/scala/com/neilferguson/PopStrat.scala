package com.neilferguson

import hex.FrameSplitter
import hex.deeplearning.DeepLearning
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.examples.h2o.DemoUtils._
import org.apache.spark.h2o.H2OContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{Genotype, GenotypeAllele}
import water.Key

import scala.collection.JavaConverters._
import scala.collection.immutable.Range.inclusive
import scala.io.Source

object PopStrat {

  private final val VARIANT_FREQUENCY_RANGE = inclusive(11, 11)

  def main(args: Array[String]): Unit = {

    val master = args(0)
    val genotypeFile = args(1)
    val panelFile = args(2)

    val sc = new SparkContext(master, "h2oTest")

    // populations to select
    val pops = Set("GBR", "ASW", "CHB")

    // TRANSFORM THE panelFile Content in the sampleID -> population map
    // containing the populations of interest (pops)
    def extract(filter: (String, String) => Boolean= (s, t) => true) = Source.fromFile(panelFile).getLines().map( line => {
      val toks = line.split("\t").toList
      toks(0) -> toks(1)
    }).toMap.filter( tup => filter(tup._1, tup._2) )

    // panel extract from file, filtering by the 2 populations
    val panel: Map[String,String] =
      extract((sampleID: String, pop: String) => pops.contains(pop))

    println("Created panel")

    val allGenotypes: RDD[Genotype] = sc.loadGenotypes(genotypeFile)

    val genotypes: RDD[Genotype] = allGenotypes.filter(genotype => {panel.contains(genotype.getSampleId)})

    // First convert the GenoTypes to our own Variant objects to conserve memory
    val variantsRDD: RDD[Variant] = genotypes.map(toVariant)

    val variantsBySampleId: RDD[(String, Iterable[Variant])] = variantsRDD.groupBy(_.sampleId)

    val sampleCount: Long = variantsBySampleId.count()

    println("Got sample count: " + sampleCount)

    val variantsByVariantId: RDD[(Int, Iterable[Variant])] = variantsRDD.groupBy(_.variantId).filter {
      case (_, variants) => variants.size == sampleCount
    }

    println("Grouped by variant ID")

    val variantsHistogram: collection.Map[Int, Int] = variantsByVariantId.map {
      case (variantId, variants) => (variantId, variants.count(_.alternateCount > 0))
    }.collectAsMap()

    println("Created variants histogram")

    val filteredVariantsBySampleId: RDD[(String, Iterable[Variant])] = variantsBySampleId.map {
      case (sampleId, variants) =>
        (sampleId, variants.filter(variant => VARIANT_FREQUENCY_RANGE.contains(variantsHistogram.getOrElse(variant.variantId, -1))))
    }

    val sortedVariantsBySampleId: RDD[(String, Array[Variant])] = filteredVariantsBySampleId.map {
      case (sampleId, variants) =>
        val variantsArray = variants.toArray.sortBy(_.variantId)
        println("Created variants array of size " + variantsArray.length)
        (sampleId, variantsArray)
    }

    // All items in the RDD should now have the same variants in the same order so we can just use the first
    // one to construct our header
    val header = StructType(Array(StructField("Region", StringType)) ++
      sortedVariantsBySampleId.first()._2.map(variant => {StructField(variant.variantId.toString, IntegerType)}))

    println("Created header")

    val rowRDD: RDD[Row] = sortedVariantsBySampleId.map {
      case (sampleId, sortedVariants) =>
        val region: Array[String] = Array(panel.getOrElse(sampleId, "Unknown"))
        val alternateCounts: Array[Int] = sortedVariants.map(_.alternateCount)
        Row.fromSeq(region ++ alternateCounts)
    }

    println("Created row RDD")

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val schemaRDD = sqlContext.applySchema(rowRDD, header)

    println("Created schema RDD")

    val h2oContext = new H2OContext(sc).start()
    import h2oContext._

    val dataFrame = h2oContext.toDataFrame(schemaRDD)

    println("Created dataframe")

    val sf = new FrameSplitter(dataFrame, Array(.7), Array("training", "test").map(Key.make), null)
    water.H2O.submitTask(sf)
    val splits = sf.getResult
    val training = splits(0)
    val test = splits(1)

    println("Column name: " + dataFrame.name(0))

    val dlParams = new DeepLearningParameters()
    dlParams._train = training
    dlParams._valid = test
    dlParams._response_column = "Region"
    dlParams._epochs = 10
    dlParams._activation = Activation.RectifierWithDropout
    dlParams._hidden = Array[Int](100,100)

    val dl = new DeepLearning(dlParams)
    val dlModel = dl.trainModel.get

    println("Built model")

    val dlPredictTableTest = dlModel.score(test)('predict)
    println("Scored model against test set")

    dlModel.score(dataFrame)('predict)
    println("Scored model against all records")

    printf(residualPlotRCode(dlPredictTableTest, "predict", test, "ArrDelay"))

  }

  def toVariant(genotype: Genotype): Variant = {
    // Intern sample IDs as they will be repeated a lot
    new Variant(genotype.getSampleId.intern(), variantId(genotype).hashCode(), alternateCount(genotype))
  }

  private def alternateCount(genotype: Genotype): Int = {
    genotype.getAlleles.asScala.count(_ != GenotypeAllele.Ref)
  }

  private def variantId(genotype: Genotype): String = {
    val name = genotype.getVariant.getContig.getContigName
    val start = genotype.getVariant.getStart
    val end = genotype.getVariant.getEnd
    s"$name:$start:$end"
  }

  case class Variant(sampleId: String, variantId: Int, alternateCount: Int)

}
