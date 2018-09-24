package myspark.mllib

import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by greghuang on 3/30/16.
  */
object DataTypeExample {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MySpark")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)

    // dense vector
    val dv : Vector = Vectors.dense(1.0, 0.0, 3.0)
    // sparse vector
    val sv : Vector = Vectors.sparse(3, Array(0,2), Array(1.0, 3.0))
    // Labeled point
    val pos = LabeledPoint(1.0, dv)
    val neg = LabeledPoint(0.0, sv)
    // Load data from LIBSVM format
    val samples : RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "src/main/resources/sample_libsvm_data.txt")
    samples.foreach(print)

    println
    // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    val dm : Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
    println(dm)
    // Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
    val sm : Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 1, 2), Array(9, 8, 6)) // there is typo on original document
    println(sm)
  }
}
