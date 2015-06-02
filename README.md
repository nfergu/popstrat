# Introduction

This is an example of population stratification analysis using "deep learning" (neural networks). It aims to produce
similar results to
[this blog post from bdgenomics.org](http://bdgenomics.org/blog/2015/02/02/scalable-genomes-clustering-with-adam-and-spark/).
The following technologies are used:

 * [ADAM](https://github.com/bigdatagenomics/adam): a genomics analysis platform and associated file formats
 * [Apache Spark](https://spark.apache.org/): a fast engine for large-scale data processing
 * [H2O](http://0xdata.com/product/): an open source predictive analytics platform
 * [Sparking Water](http://0xdata.com/product/sparkling-water/): integration of H2O with Apache Spark

The example consists of a single Scala class: `PopStrat`.

# Prerequisites

Before building and running PopStrat ensure you have the
[http://www.oracle.com/technetwork/java/javase/downloads/index.html](Java JDK) installed, version 7 or later.

# Building

To build from source first [download and install Maven](http://maven.apache.org/download.cgi).
Then at the command line type:

```
mvn clean package
```

This will build a JAR (target/uber-popstrat-0.1-SNAPSHOT.jar) containing the `PopStrat` class,
as well as all of its dependencies

# Running

First [download Spark version 1.2.0](http://spark.apache.org/downloads.html) and unpack it on your machine.

Next you'll need to get some genomics data. From 1000 genomes, download
[the chromosome 22 genotype data](ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz) and
[the panel file](ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel).

Unzip the genotype data before continuing. This will require around 10GB of disk space.

To speed up execution and save disk space you can convert the genotype VCF file to [ADAM](https://github.com/bigdatagenomics/adam)
format (using the [ADAM](https://github.com/bigdatagenomics/adam) `transform` command) if you wish. However
this will take some time up-front. Both ADAM and VCF formats are supported.

Next run the following command:

```
YOUR_SPARK_HOME/bin/spark-submit --class "com.neilferguson.PopStrat" --master local[6] --driver-memory 6G target/uber-popstrat-0.1-SNAPSHOT.jar <genotypesfile> <panelfile>
```

Replacing <genotypesfile> with the path to your genotype data file (ADAM or VCF), and <panelfile> with the panel file
from 1000 genomes.

This runs PopStrat using a local (in-process) Spark master with 6 cores and 6GB of RAM. You can run against a different
Spark cluster by modifying the options in the above command line. See the
[Spark documentation](https://spark.apache.org/docs/1.2.0/submitting-applications.html) for further details.

Using the above data PopStrat may take up to 2-3 hours to run, depending on hardware. When it is finished you should
see some output like the following:

```

```

This is a [confusion matrix](http://en.wikipedia.org/wiki/Confusion_matrix) which shows the predicted versus the actual
populations. All being well, you should see accuracy of more than 99% (only one or two predictions should be incorrect).

# Code

The code is fairly straightforward and should be easy to follow. It follows the following high level flow:

 1. Load the genotype and panel data from the specified files
 2. Filter out those samples that aren't in the populations we are trying to predict
 3. Filter out variants that are missing from some samples
 4. Reduce the number of dimensions in the data by filtering to a (fairly arbitrary) subset of variants
 5. Create a Spark `SchemaRDD` with a each column representing a variant and each row representing a sample
 6. Convert the `SchemaRDD` to an H2O data frame.
 7. Convert the data frame into 50% training data and 50% test data
 8. Set the parameters for the deep learning model and train the model
 9. Score the entire data set (training and test data) against the model

# Credits

Thanks to the folks at [Big Data Genomics] for the
[original blog post](http://bdgenomics.org/blog/2015/02/02/scalable-genomes-clustering-with-adam-and-spark/)
that inspired this.