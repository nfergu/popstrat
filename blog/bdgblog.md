# Genomic Analysis Using ADAM, Spark and Deep Learning

## Introduction

In this post we will apply a technique known as "[deep learning](https://en.wikipedia.org/wiki/Deep_learning)"
using [artifical neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
to predict which population group an individual belongs to based on their genome.

This is a follow-up to an earlier post:
[Scalable Genomes Clustering With ADAM and Spark](http://bdgenomics.org/blog/2015/02/02/scalable-genomes-clustering-with-adam-and-spark/)
and attempts to replicate the results of that post. However we will use a different machine learning technique;
where the original post used [k-means clustering(https://en.wikipedia.org/wiki/K-means_clustering) we will use
deep learning.

We will use [ADAM](https://github.com/bigdatagenomics/adam) and [Apache Spark](https://spark.apache.org/) in
combination with [H2O](http://0xdata.com/product/), an open source predictive analytics platform, and
[Sparking Water](http://0xdata.com/product/sparkling-water/), which integrates H2O with Spark.

## Code

In this section we'll dive straight into the code. If you'd rather get something working before looking at the code
you can skip to the "Building and Running" section.

The complete Scala code for this example can be found in
[The PopStrat.scala class on GitHub](https://github.com/nfergu/popstrat/blob/master/src/main/scala/com/neilferguson/PopStrat.scala)
and we'll refer to sections of the code here. Basic familiarity with Scala and
[Apache Spark](https://spark.apache.org/) is assumed.

### Setting-up

The first thing we need to do is read the names of the Genotype and Panel files that are passed into our program.
The Genotype file contains data about a set of individuals (referred to here as "samples") and their genetic
variation. The Panel file lists the population group (or "region") for each sample in the Genotype file; this is
the thing that we will try and predict.

```scala
val genotypeFile = args(0)
val panelFile = args(1)
```

Next we set-up our Spark Context. Our program permits the Spark master to be specified as one of its arguments.
This is useful when running from an IDE, but is omitted when running from the ```spark-submit``` script (see below).

```scala
val master = if (args.length > 2) Some(args(2)) else None
val conf = new SparkConf().setAppName("PopStrat")
master.foreach(conf.setMaster)
val sc = new SparkContext(conf)
```

Next we declare a set called ```populations``` which contains all of the population groups that we're interested
in predicting. We then read the Panel file into a Map, filtering it based on the population groups in the
```populations``` set. The format of the panel file is described [here](http://www.1000genomes.org/faq/what-panel-file),
but it's very simple, containing the sample ID in the first column and the population group in the second.

```scala
val populations = Set("GBR", "ASW", "CHB")
def extract(file: String, filter: (String, String) => Boolean): Map[String,String] = {
  Source.fromFile(file).getLines().map(line => {
    val tokens = line.split("\t").toList
    tokens(0) -> tokens(1)
  }).toMap.filter(tuple => filter(tuple._1, tuple._2))
}
val panel: Map[String,String] = extract(panelFile, (sampleID: String, pop: String) => populations.contains(pop))
```

### Preparing the Genomics Data

Next we use [ADAM](https://github.com/bigdatagenomics/adam) to read our genotype data into a Spark RDD. Since we've
imported ```ADAMContext._``` at the top of our class this is simply a matter of calling `loadGenotypes` on the
Spark Context. Then we filter the genotype data to contain only samples that are in the population groups that we're
interested in.

```scala
val allGenotypes: RDD[Genotype] = sc.loadGenotypes(genotypeFile)
val genotypes: RDD[Genotype] = allGenotypes.filter(genotype => {panel.contains(genotype.getSampleId)})
```

Next we convert the ADAM ```Genotype``` objects into our own ```SampleVariant``` objects containing just the data
we need for further processing: the sample ID, which uniquely identifies a particular sample, a variant ID, which
uniquely identifies a particular genetic variant, and a count of alternate
[alleles](http://www.snpedia.com/index.php/Allele), where the sample differs from the
reference genome. These variations will help us to classify individuals according to their population group.

```scala
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
```

Next we count the total number of samples (individuals) in the data. We then group the data by variant ID and filter
out those variants that do not appear in all of the samples. This makes processing of the data simpler, and since
we have a very large number of variants in the data (up to 30 million, depending on the exact data set) filtering out
a small number will not make a significant difference to the results. In fact, in the next step we'll reduce the
number of variants even further.

```scala
val variantsBySampleId: RDD[(String, Iterable[SampleVariant])] = variantsRDD.groupBy(_.sampleId)
val sampleCount: Long = variantsBySampleId.count()
println("Found " + sampleCount + " samples")
val variantsByVariantId: RDD[(Int, Iterable[SampleVariant])] = variantsRDD.groupBy(_.variantId).filter {
  case (_, sampleVariants) => sampleVariants.size == sampleCount
}
```

When we train our machine learning model each variant will be treated as a
"[feature](https://en.wikipedia.org/wiki/Feature_(machine_learning))" that is used to train the model.
Since it can be difficult to train machine learning models with very large numbers of features in the data
(particularly if the number of samples is relatively small) we first need to try and reduce the number of variants
in the data.

To do this we first compute the frequency with which alternate alleles have occurred for each variant. We then
filter the variants down to just those that appear within a certain frequency range. In this case we've chosen a
fairly arbitrary frequency of 11. This was chosen through experimentation as a value that leaves around 3,000 variants
in the data set we are using.

There are more structured approaches to
[dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), like
[principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) that we could have
employed, but this technique seems to work well enough for this example.

```scala
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
```

### Creating the Training Data

To train our model we need our data to be in tabular form where each row represents a single sample, and each
column represents a specific variant. The table also contains a column for the population group or "Region", which is
the thing that we are trying to predict.

Ultimately, in order for our data to be consumed by H2O we need it to end up in an H2O `DataFrame` object. Currently
the best way to do this in Spark seems to be to convert our data to an RDD of Spark SQL
[Row](http://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.sql.Row] objects, and then this can
automatically be converted to an H2O DataFrame.

To achieve this we first need to group the data by sample ID, and then sort the variants for each sample in a
consistent manner (by variant ID). We can then create a header row for our table, containing the Region column,
the sample ID, and all of the variants. We then create an RDD of type `Row` for each sample.

```scala
val sortedVariantsBySampleId: RDD[(String, Array[SampleVariant])] = filteredVariantsBySampleId.map {
  case (sampleId, variants) =>
    (sampleId, variants.toArray.sortBy(_.variantId))
}
val header = StructType(Array(StructField("Region", StringType)) ++
  sortedVariantsBySampleId.first()._2.map(variant => {StructField(variant.variantId.toString, IntegerType)}))
val rowRDD: RDD[Row] = sortedVariantsBySampleId.map {
  case (sampleId, sortedVariants) =>
    val region: Array[String] = Array(panel.getOrElse(sampleId, "Unknown"))
    val alternateCounts: Array[Int] = sortedVariants.map(_.alternateCount)
    Row.fromSeq(region ++ alternateCounts)
}
```

As mentioned above, once we have our RDD of `Row` objects we can then convert these automatically to an H2O
DataFrame using Sparking Water (H2O's Spark integration).

```scala
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val schemaRDD = sqlContext.applySchema(rowRDD, header)
val h2oContext = new H2OContext(sc).start()
import h2oContext._
val dataFrame = h2oContext.toDataFrame(schemaRDD)
```

Now that we have a DataFrame we want to split it into the training data, that we'll use to train our model, and a
[test set](https://en.wikipedia.org/wiki/Test_set) that we'll use to ensure that
[overfitting](https://en.wikipedia.org/wiki/Overfitting) has not occurred.

We will also create a "validation" set which performs a similar purpose to the test set, in that it will be used to
validate the strength of our model as it is being built, while avoiding overfitting. However, when training a neural
network we typically keep the validation set distinct from the test set, to enable us to learn
[hyper-parameters](http://colinraffel.com/wiki/neural_network_hyperparameters) for the model.
See [chapter 3 of Michael Nielsen's "Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/chap3.html)
for more details on this.

H2O comes with a class called `FrameSplitter`, so splitting the data is simply a matter of calling creating one
of those and letting it split the data set.

```scala
val frameSplitter = new FrameSplitter(dataFrame, Array(.5, .3), Array("training", "test", "validation").map(Key.make), null)
water.H2O.submitTask(frameSplitter)
val splits = frameSplitter.getResult
val training = splits(0)
val validation = splits(2)
```

### Training the Model

Next we need to set the parameters for our deep learning model. We specify the training and validation data sets,
as well as the column in the data that contains the item we are trying to predict (in this case the Region).
We also set some [hyper-parameters](http://colinraffel.com/wiki/neural_network_hyperparameters) that affect the way
the model learns. We won't go into detail about these here, but you can read more in the
[H2O documentation](http://docs.h2o.ai/h2oclassic/datascience/deeplearning.html). These parameters have been
chosen through experimentation - however H2O provides methods for
[automatically tuning hyper-parameters](http://learn.h2o.ai/content/hands-on_training/deep_learning.html) so
it may be possible to achieve better results by employing one of these methods.

```scala
val deepLearningParameters = new DeepLearningParameters()
deepLearningParameters._train = training
deepLearningParameters._valid = validation
deepLearningParameters._response_column = "Region"
deepLearningParameters._epochs = 10
deepLearningParameters._activation = Activation.RectifierWithDropout
deepLearningParameters._hidden = Array[Int](100,100)
```

Finally, we're ready to train our deep learning model! Now that we've set everything up this is easy:
we simply create a H2O `DeepLearning` object and call `trainModel` on it.

```scala
val deepLearning = new DeepLearning(deepLearningParameters)
val deepLearningModel = deepLearning.trainModel.get
```

Having trained our model in the previous step we now need to check how well it predicts the population
groups in our data set. To do this we "score" our entire data set (including training, test, and validation data)
against our model:

```scala
deepLearningModel.score(dataFrame)('predict)
```

This final step will print a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) which shows how
well our model predicts our population groups. All being well the confusion matrix should look something like this:

```
Confusion Matrix (vertical: actual; across: predicted):
       ASW CHB GBR  Error      Rate
   ASW  60   1   0 0.0164 =  1 / 61
   CHB   0 103   0 0.0000 = 0 / 103
   GBR   0   1  90 0.0110 =  1 / 91
Totals  60 105  90 0.0078 = 2 / 255
```

This tells us that the model has correctly predicted 253 out of 255 population groups correctly (an accuracy of
more than 99%). Nice!

## Building and Running

### Prerequisites

Before building and running the example ensure you have version 7 or later of the
[Java JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html) installed.

### Building

First clone the GitHub repo at [https://github.com/nfergu/popstrat].

Then [download and install Maven](http://maven.apache.org/download.cgi). Then at the command line type:

```
mvn clean package
```

This will build a JAR (target/uber-popstrat-0.1-SNAPSHOT.jar) containing the `PopStrat` class,
as well as all of its dependencies.

### Running

First [download Spark version 1.2.0](http://spark.apache.org/downloads.html) and unpack it on your machine.

Next you'll need to get some genomics data. Go to your
[nearest mirror of the 1000 genomes FTP site](http://www.1000genomes.org/data#DataAccess).
From the `release/20130502/` directory download
the `ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz` file and
the `integrated_call_samples_v3.20130502.ALL.panel` file. The first file file is the genotype data for chromosome 22,
and the second file is the panel file, which describes the population group for each sample in the genotype data.

Unzip the genotype data before continuing. This will require around 10GB of disk space.

To speed up execution and save disk space you can convert the genotype VCF file to [ADAM](https://github.com/bigdatagenomics/adam)
format (using the [ADAM](https://github.com/bigdatagenomics/adam) `transform` command) if you wish. However
this will take some time up-front. Both ADAM and VCF formats are supported.

Next run the following command:

```
YOUR_SPARK_HOME/bin/spark-submit --class "com.neilferguson.PopStrat" --master local[6] --driver-memory 6G target/uber-popstrat-0.1-SNAPSHOT.jar <genotypesfile> <panelfile>
```

Replacing &lt;genotypesfile&gt; with the path to your genotype data file (ADAM or VCF), and &lt;panelfile&gt; with the panel file
from 1000 genomes.

This runs PopStrat using a local (in-process) Spark master with 6 cores and 6GB of RAM. You can run against a different
Spark cluster by modifying the options in the above command line. See the
[Spark documentation](https://spark.apache.org/docs/1.2.0/submitting-applications.html) for further details.

Using the above data PopStrat may take up to 2-3 hours to run, depending on hardware. When it is finished you should
see a [confusion matrix](http://en.wikipedia.org/wiki/Confusion_matrix) which shows the predicted versus the actual
populations. If all has gone well this should show an accuracy of more than 99%.
See the "Code" section above for more details on what exactly you should expect to see.

## Conclusion

In this post we have seen how to combine ADAM and Apache Spark with H2O's deep learning capabilities to predict
population groups based on genomic data. Our results show that we can predict these very well, with over
99% accuracy. Our choice of technologies makes for a relatively straightforward implementation, and will likely make
for a very scalable solution.

Future work could involve validating the scalability of our solution on more hardware, trying to predict a wider
range of population groups (currently we only predict 3 groups), and tuning the deep learning hyper-parameters to
achieve even better accuracy.
