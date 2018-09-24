package myspark.ml.feature

import org.apache.spark.ml.feature.{Word2Vec, IDF, HashingTF, Tokenizer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by GregHuang on 2/17/16.
 */
object Word2VecSample {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MySpark")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)
    val sqlCtx = new SQLContext(sc)

    val documentDF = sqlCtx.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)
    result.select("result").foreach(println)
  }
}
