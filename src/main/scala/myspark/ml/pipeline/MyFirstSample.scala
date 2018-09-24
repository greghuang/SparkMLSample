package myspark.ml.pipeline

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by GregHuang on 2/15/16.
 */
object MyFirstSample {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MySpark")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)

    val sqlCtx = new SQLContext(sc)

    val training = sqlCtx.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label","features")

    val lr = new LogisticRegression()
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    lr.setMaxIter(10).setRegParam(0.01)

    val model1 = lr.fit(training)
    println("Model 1 was fit using parameter: " + model1.parent.extractParamMap())

    val paramMap = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter -> 30).put(lr.regParam -> 0.1, lr.threshold -> 0.55)

    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // Change output column name
    val paramMapCombined = paramMap ++ paramMap2

    val model2 = lr.fit(training, paramMapCombined)

    // Prepare test data.
    val test = sqlCtx.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")


    model2.transform(test)
    .select("features", "label", "myProbability", "prediction")
    .collect()
    .foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) => println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }
  }
}
