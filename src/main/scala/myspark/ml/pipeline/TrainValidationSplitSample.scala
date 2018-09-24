package myspark.ml.pipeline

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by GregHuang on 2/17/16.
 */
object TrainValidationSplitSample {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf(false)
      .setMaster("local[*]")
      .setAppName("MySpark")
      .set("spark.driver.port", "7777")
      .set("spark.driver.host", "localhost")

    val sc = new SparkContext(sparkConf)

    val sqlCtx = new SQLContext(sc)

    // Prepare training and test data.
    val data = sqlCtx.read.format("libsvm").load("/Users/GregHuang/Project/Spark/spark/data/mllib/sample_libsvm_data.txt")

    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

    val lr = new LinearRegression()

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // In this case the estimator is simply the linear regression.
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    model.transform(test)
      .select("features", "label", "prediction")
      .show()

    //Note: This sample will raise assertion exception due to ill-condition problem
  }
}
