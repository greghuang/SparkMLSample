package myspark.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import scopt.OptionParser
/**
  * Created by greghuang on 4/10/16.
  */
object MultivariateSummarier {
  case class Params(input: String = "src/main/resources/sample_linear_regression_data.txt")
    extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MultivariateSummarizer"){
      head("MultivariateSummarizer: an example app for MultivariateOnlineSummarizer")
      opt[String]("input")
        .text(s"Input path to labeled examples in LIBSVM format, default: ${defaultParams.input}")
        .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.MultivariateSummarizer \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --input data/mllib/sample_linear_regression_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params : Params): Unit = {
    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MultivariateSummarier")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)

    val examples = MLUtils.loadLibSVMFile(sc, params.input).cache()

    println(s"Summary of data file ${params.input}")
    println(s"${examples.count()} data samples")

    sc.stop()
  }
}
