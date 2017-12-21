import org.apache.spark.{SparkConf, SparkContext}
import RecommenderEngine.{readClusters, readMovieFeatures, recommender_learner}
import Utilities.readAndFilter

object Task {

    /**
      * Main method
      * args{0} -> ratings file
      * args{1} -> testing file
      * args{2} -> optional (give anything to run in cross validated mode. DO NOT USE UNLESS YOU WANT TO TUNE A PARAMETER.
      * @param args input arguments
      */
    def main(args: Array[String]): Unit ={
        var crossValidate = false
        if(args.length == 3){
            crossValidate = true
            println("Cross-validating")
        }
        val sparkConf = new SparkConf().setAppName("Assignment 2 - Case 1").setMaster("local[*]")
        val sparkContext = new SparkContext(sparkConf)
        val USER_NEIGHBORHOOD_THRESHOLD = 25
        val ITEM_NEIGHBORHOOD_THRESHOLD = 200
        val WEIGHTAGE = 0.5

        if(!crossValidate){
            //Actual run for the recommender task.
            val readData = readAndFilter(args{0}, args{1}, sparkContext)
            val ratingsFiltered = readData._1
            val ratingsTest = readData._3
            val validation = readData._2
            println(args.mkString(","))
            val cosineSim = readMovieFeatures(args{2}, args{3})
            val clusters = readClusters(args{4},cosineSim._2)
            val rmse = recommender_learner(ratingsFiltered, validation, ratingsTest, USER_NEIGHBORHOOD_THRESHOLD, ITEM_NEIGHBORHOOD_THRESHOLD, sparkContext, cosineSim._1, Utilities.reverseIndex(cosineSim._2), clusters._1, clusters._2, WEIGHTAGE)
            println("RMSE: "+rmse)
        }else{

            //Cross validation mode - does 5 fold. Not to give too high of a value.
            val kFold: Int = 5
            var rmseSummer: Double = 0.0
            val splits: Array[scala.collection.mutable.Set[(Int, Int, Double)]] = Utilities.getSplits(args{0}, kFold)
            println(splits{1}.size)

            val cosineSim = readMovieFeatures(args{2}, args{3})
            for(i <- 0 until kFold){
                val readData = Utilities.trainTestSplit(splits, i, sparkContext)
                val ratingsFiltered = readData._1
                val ratingsTest = readData._3
                val validation = readData._2
                val clusters = readClusters(args{4},cosineSim._2)
                val reverseIndex = Utilities.reverseIndex(cosineSim._2)
                val rmse = recommender_learner(ratingsFiltered, validation, ratingsTest, USER_NEIGHBORHOOD_THRESHOLD, ITEM_NEIGHBORHOOD_THRESHOLD, sparkContext, cosineSim._1, reverseIndex, clusters._1, clusters._2, WEIGHTAGE)
                println("part",i, rmse)
                rmseSummer += rmse
            }
            println("Average: "+(rmseSummer/kFold))

        }
    }

}
