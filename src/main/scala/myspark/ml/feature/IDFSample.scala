package myspark.ml.feature

import org.apache.spark.ml.feature.{IDF, HashingTF, Tokenizer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by GregHuang on 2/17/16.
 */
object IDFSample {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MySpark")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)
    val sqlCtx = new SQLContext(sc)

    val sentenceData = sqlCtx.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("sentence", "features", "label").take(3).foreach(println)
  }

}
