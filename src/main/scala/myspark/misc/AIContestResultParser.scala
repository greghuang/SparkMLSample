package myspark.misc

import org.apache.spark
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by greghuang on 2018/9/24.
  */
object AIContestResultParser {
  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MySpark")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)

    val sqlCtx = new SQLContext(sc)
    import sqlCtx.implicits._

    val df = sqlCtx.read.json("/Users/greghuang/Documents/Programming/Spark/SparkMLSample/data/week1_all_result.txt")
    df.printSchema()
    df.show()

    val df2 = df.
      withColumn("TrendHearts", $"trendHearts".getField("total").cast("Int")).
      withColumn("FormulaTrend", $"formulaTrend".getField("total").cast("Int")).
      withColumn("Bonus", $"bonus".getField("total").cast("Int")).
      select("rank", "playerNumber", "TrendHearts", "FormulaTrend", "Bonus", "total")

    df2.show()

    df2.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").save("/Users/greghuang/Documents/Programming/Spark/SparkMLSample/data/week1_all_result")

  }
}
