import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD

import collection.mutable.Set
import scala.io.Source
import scala.util.Random
import util.control.Breaks._


object Utilities {
    /********************************************************
      * HELPER METHODS
      ********************************************************/

    /**
      * Calculates the RMSE for the learned recommender by comparing the predictions with the validation scores.
      * The absolute difference between the predicted rating and the actual rating is grouped and printed.
      * This gives you an idea of how good the recommender is.
      * @param predictions predicted by the user CF recommender
      * @param validation ground truth
      * @return RMSE score
      */
    def measures(predictions: RDD[((Int, Int), Double)], validation: RDD[((Int, Int), Double)]): Double ={
        val ratesAndPreds = validation.join(predictions)

        val RMSE = ratesAndPreds.map { case ((_,_), (r1, r2)) =>
            if (r2 != None) {
                val err = (r1 - math.round(r2))
                err * err
            }else 0
        }.mean()

        //Grouping them with respect to absolute differences
        val grouped = ratesAndPreds.map{
            case ((user, product), (actual, predicted)) =>
                val absDifference = Math.abs(predicted - actual)
                if(absDifference >= 0 && absDifference < 1){
                    (">=0 and <1", 1)
                }else if(absDifference >= 1 && absDifference < 2){
                    (">=1 and <2", 1)
                }else if(absDifference >= 2 && absDifference < 3){
                    (">=2 and <3", 1)
                }else if(absDifference >= 3 && absDifference < 4){
                    (">=3 and <4", 1)
                }else{
                    (">=4", 1)
                }
        }.reduceByKey((x,y)=> x+y).collectAsMap()

        //Writing into File.
        val predsIntoFile = ratesAndPreds.map(x => ((x._1._1, x._1._2), x._2._2)).sortByKey().collect()
        writeIntoFile(predsIntoFile,"Sriram_Baskaran_result.txt")

        println(">=0 and <1: "+grouped.getOrElse(">=0 and <1",0))
        println(">=1 and <2: "+grouped.getOrElse(">=1 and <2",0))
        println(">=2 and <3: "+grouped.getOrElse(">=2 and <3",0))
        println(">=3 and <4: "+grouped.getOrElse(">=3 and <4",0))
        println(">=4: "+grouped.getOrElse(">=4",0))
        println("RMSE = " + math.sqrt(RMSE))

        RMSE
    }

    /**
      * Write the predictions into a file as per specified format.
      * @param predsIntoFile
      * @param filename
      */

    def writeIntoFile(predsIntoFile: Array[((Int, Int), Double)], filename: String): Unit= {
        val pw:PrintWriter = new PrintWriter(new File(filename))
        pw.write("UserID,MovieId,Pred_rating\n")
        for(pred <- predsIntoFile){
            pw.write(pred._1._1+","+pred._1._2+","+pred._2+"\n")
        }
        pw.close()
    }

    /***********************************************
      * Methods to do Cross-validation
      **********************************************/

    def getSplits(ratingsFile :String, kFold:Int): Array[scala.collection.mutable.Set[(Int, Int, Double)]] = {

        val ratingLines = Source.fromFile(ratingsFile).getLines().toList
        val ratingsShuffled = Random.shuffle(ratingLines.slice(1,ratingLines.size))

        var splits: Array[scala.collection.mutable.Set[(Int, Int, Double)]] = new Array(kFold)
        var first = true
        var i = 0
        for (line <- ratingsShuffled) {
            breakable{
                if(first){
                    first = false
                    break
                }
            }
            val tokens = line.split(",")
            val user = tokens {0}.toInt
            val movie = tokens {1}.toInt
            val rating = tokens {2}.toDouble
            if(splits{i%kFold} == null){
                splits{i%kFold} = scala.collection.mutable.Set()
            }
            splits{i%kFold}.add((user, movie, rating))
            i += 1
        }

        splits
    }

    /**
      * Create a training and test set from the partitions
      * @param splits the created partitions
      * @param part the part to be used as dev set
      * @param sc spark context
      * @return
      */

    def trainTestSplit(splits: Array[scala.collection.mutable.Set[(Int, Int, Double)]], part: Int, sc: SparkContext): (RDD[(Int, Int, Double)], RDD[(Int, Int, Double)], RDD[(Int, Int)]) ={


        val validation: scala.collection.mutable.Set[(Int, Int, Double)] = splits{part}
        val indices = splits.indices.filter(x => x!=part)
        val ratings: scala.collection.mutable.Set[(Int, Int, Double)] = indices.map(x => splits{x}).reduce((x,y) => x++y)
        val testing: scala.collection.mutable.Set[(Int, Int)]  = validation.map(x => (x._1, x._2))

        return (sc.parallelize(ratings.toSeq), sc.parallelize(validation.toSeq), sc.parallelize(testing.toSeq))


    }

    /***********************************************
      * Reads input file and splits it into training, test and validation parts. Creates an RDD for the data and returns
      * the RDD. Spark context is taken as input for this.
      *
      *
      * @param ratingsFile ratings file name
      * @param testFile dev set file name
      * @param sc spark context
      * @return
      **********************************************/

    def readAndFilter(ratingsFile :String, testFile :String, sc: SparkContext): (RDD[(Int, Int, Double)], RDD[(Int, Int, Double)], RDD[(Int, Int)]) = {

        val ratingLines = Source.fromFile(ratingsFile).getLines()
        val testingLines = Source.fromFile(testFile).getLines()


        val ratings: Set[(Int, Int, Double)] = Set()
        val testing: Set[(Int, Int)] = Set()
        val validation: Set[(Int, Int, Double)] = Set()
        var first: Boolean = true

        for (line <- testingLines) {
            breakable{
                if(first){
                    first = false
                    break
                }
                val tokens = line.split(",")
                val user = tokens{0}.toInt
                val movie = tokens{1}.toInt
                testing.add((user, movie))
            }
        }

        first = true
        for (line <- ratingLines) {
            breakable {
                if (first) {
                    first = false
                    break
                }
                val tokens = line.split(",")
                val user = tokens {0}.toInt
                val movie = tokens {1}.toInt
                val rating = tokens {2}.toDouble
                if (!testing.contains((user, movie))) {
                    ratings.add((user, movie, rating))
                }else{
                    validation.add((user, movie, rating))
                }
            }
        }

        (sc.parallelize(ratings.toSeq), sc.parallelize(validation.toSeq), sc.parallelize(testing.toSeq))
    }

    def reverseIndex(labels: List[Int]): Map[Int, Int] ={
        val iterator = Array.range(0,labels.size)
        iterator.map(x => (labels{x}, x)).toMap
    }

    def add(a: Array[Int], b: Array[Int]): Array[Int] = {
        a.zip(b).map { case (x, y) => x + y }
    }
}
