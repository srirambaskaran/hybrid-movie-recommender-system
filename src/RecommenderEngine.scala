import java.util.Calendar

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.io.Source
import scala.util.Random


object RecommenderEngine {

    final val STANDARD_RATING_MEAN = 4.0
    final val STANDARD_RATING_STDDEV = 0.5

    /**
      * Read cluster file. Clustering is done using Scipy.
      * @param clustersFile input cluster file
      * @param labels a map from index to the movie id
      * @return
      */
    def readClusters(clustersFile: String, labels: List[Int]): (Map[Int, List[Int]], Map[Int, Int]) = {
        val clusterFileLines = Source.fromFile(clustersFile).getLines()
        val clusterLinesSplit = clusterFileLines.map(x => x.split(" ")).toList

        val clusterMemberships = clusterLinesSplit.map(x => x{1}.toInt -> labels(x{0}.toInt)).groupBy(_._1).map{ case (k,v) => (k , v.map(_._2))}
        val pointClusterMap = clusterLinesSplit.map(x => (labels(x{0}.toInt),x{1}.toInt)).toMap
        (clusterMemberships, pointClusterMap)
    }

    /**
      * calculates cosine similarity
      * @param a vector
      * @param b vector
      * @return cosine similarity
      */

    def cosine(a: Array[Double], b: Array[Double]): Double = {
        Array.range(0,a.length).map(x => a{x}*b{x}).sum / (math.sqrt(a.map(x => x*x).sum) * math.sqrt(b.map(x => x*x).sum))
    }

    /**
      * calculates jaccard similarity
      * @param a vector
      * @param b vector
      * @return jaccard similarity
      */

    def jaccard(a: Array[Int], b: Array[Int]): Double = {
        val iterator = Array.range(0,a.length)
        val intersection = iterator.count(x => a{x} == b{x} && a{x} == 1)
        val union = a.sum + b.sum - intersection
        intersection.toDouble/union
    }

    /**
      * Reading input file to get features
      * @param labelsFile a map from index to the movie id
      * @param encodedDataFile one hot encoded movie file. encoding is done for each feature if it exists.
      * @return
      */

    def readMovieFeatures(labelsFile: String, encodedDataFile: String): (List[Array[Int]], List[Int])= {
        val encodedDataLines = Source.fromFile(encodedDataFile).getLines()
        val labels = Source.fromFile(labelsFile).getLines().map(x => x.toInt).toList
        val encoded = encodedDataLines.map(x => x.split(" ").map(x => x.toInt)).toList
        (encoded, labels)
    }


    /**
      * Imputation for ratings for cold starts. Pick a value from normal distribution defined by
      * STANDARD_RATING_MEAN and STANDARD_RATING_STDDEV
      * @return
      */
    def standardRatingDistrubution(): Double ={
        Random.nextGaussian() * STANDARD_RATING_STDDEV + STANDARD_RATING_MEAN
    }

    /**
      * Learns a User Collaborative Filtering recommender system using pearson correlation.
      * The function pre-calculates additional features like user history, item history and pearson correlation.
      * User CF is calculated since the data contains lesser number of users when comapred to items.
      * Co-rated items between different users are calculated based on a cartesian product of user-keyed map of items.
      * pearson() function calculates the correlation between the two users.
      * userCFPredict() is called with the learned correlation, user and item related histories to get a predicted score.
      * The predicted scores are validated with the ground-truth and RMSE is calculated.
      * The predicted scores are also grouped based on rating ranges.
      *
      * @param ratingsFiltered RDD of user-item rating after removing all testing data.
      * @param validation RDD of user-item rating from testing data
      * @param ratingsTest RDD of user-item for which rating needs to be predicted
      * @param USER_NEIGHBORHOOD_THRESHOLD Chosen neighbourhood threshold
      * @param sparkContext Spark context for Map-Reduce operations inside this function.
      * @return
      */
    def recommender_learner(ratingsFiltered: RDD[(Int, Int, Double)],
                            validation: RDD[(Int, Int, Double)],
                            ratingsTest: RDD[(Int, Int)],
                            USER_NEIGHBORHOOD_THRESHOLD: Int,
                            ITEM_NEIGHBORHOOD_THRESHOLD: Int,
                            sparkContext: SparkContext,
                            encoded: List[Array[Int]],
                            labels: Map[Int, Int],
                            clusterMemberships: Map[Int, List[Int]],
                            pointClusterMap: Map[Int, Int],
                            weightage: Double): Double = {

        //Storing time for Bookkeeping purposes.
        val start = Calendar.getInstance().getTimeInMillis

        //Calculating histories form given data.
        val userHistoryRDD = ratingsFiltered.map(x => (x._1, Map(x._2 -> x._3))).reduceByKey((x,y) => x++y)
        val userHistory = userHistoryRDD.collectAsMap()
        val userProfile = getUserFeatures(userHistory, encoded, labels)
        val userHistorySummed = userHistoryRDD.map(x => (x._1, x._2.values.sum)).collectAsMap()
        val itemHistory = ratingsFiltered.map(x => (x._2, Map(x._1 -> x._3))).reduceByKey((x,y) => x++y).collectAsMap()

        //Calculating corated items
        val userKeyed = ratingsFiltered.map(x => (x._1, Map(x._2 -> x._3))).reduceByKey((map1, map2) => map1 ++ map2)
        //Broadcasted and cached to support faster processing of RDDs.
        userKeyed.cache()

        //Cartesian product to calculate corated items.
        val product = userKeyed.cartesian(userKeyed)
                .map{
                    case ((user1, map1), (user2, map2)) =>
                        val keys = map1.keys.toSet.intersect(map2.keys.toSet)
                        val corated: Set[(Int, (Double, Double))] = keys.map{
                            case (key)=>
                                (key, (map1(key), map2(key)))
                        }
                        (user1, user2, corated)
                }

        //Broadcasted and cached to support faster processing
        product.cache()

        val userCFPearson = product.map(x => (x._1, Map((x._2, pearson(x._3))))).reduceByKey((map1, map2) => map1 ++ map2).collectAsMap()
        sparkContext.broadcast(userCFPearson)

        val predictions = ratingsTest
            .map(x => ((x._1, x._2) , userCFPredict(x._1, x._2, itemHistory, userHistory, userHistorySummed, userCFPearson, USER_NEIGHBORHOOD_THRESHOLD, ITEM_NEIGHBORHOOD_THRESHOLD, encoded, labels, clusterMemberships, pointClusterMap, weightage, userProfile)))
            .map {
                case ((user, item), pred) =>
                    if (pred > 5)
                        ((user, item), 5.0)
                    else if(pred < 0){
                        ((user, item), 0.0)
                    }else{
                        ((user, item), pred)
                    }
            }

        val rmse = Utilities.measures(predictions, validation.map(x => ((x._1, x._2), x._3)))
        val end = Calendar.getInstance().getTimeInMillis
        val difference = Calendar.getInstance()
        difference.setTimeInMillis(end - start)
        println("The total execution time taken is "+difference.getTime.getTime/1000.0+" secs.")

        rmse
    }


    /**
      * User based CF algorithm.
      * rating(user, item) = average(user) + variance by users who have rated the item and is in neighborhood(user)
      * If item and user is not seen (coldstart)
      *     Using content based similarity index for the chosen item defined by cosine similarity.
      *     or
      *     Use Imputation. pick a random value from gaussian with a given mean and standard deviation.
      *     The mean and standard deviation is calculated from ratings in training data (4.0 +/- 0.5).
      * If item is not seen but user has rated other items
      *     Use user's average.
      *
      * Use user's average whenever the norm or if none of user's neighbourhood has rated the item to predict.
      * @param user user for which rating needs to be predicted
      * @param itemToPredict item for which rating needs to be predicted
      * @param itemHistory map of history of users who have rated each item
      * @param userHistory map history of items rated by each user
      * @param userHistorySummed a summed version of userHistory to avoid repeated sum calculation
      * @param userCFPearson pearson correlation matrix
      * @param userNeighborThreshold neighbors to consider for prediction
      * @return
      */
    def userCFPredict(user: Int,
                      itemToPredict: Int,
                      itemHistory: scala.collection.Map[Int, scala.collection.Map[Int, Double]],
                      userHistory: scala.collection.Map[Int, scala.collection.Map[Int, Double]],
                      userHistorySummed: scala.collection.Map[Int, Double],
                      userCFPearson: scala.collection.Map[Int, scala.Predef.Map[Int, Double]],
                      userNeighborThreshold: Int,
                      itemNeighborThreshold: Int,
                      encoded: List[Array[Int]],
                      labels: Map[Int, Int],
                      clusterMemberships: Map[Int, List[Int]],
                      pointClusterMap: Map[Int, Int],
                      weightage: Double,
                      userProfile: Map[Int, Array[Double]]): Double ={

        if(!itemHistory.contains(itemToPredict)){
            if(!userHistory.contains(user))
                standardRatingDistrubution()
            else if (labels.contains(itemToPredict)){
                //Using content based similarity index for the chosen item defined by cosine similarity.
                val userRated = userHistory(user)
                val cluster = pointClusterMap(itemToPredict)
                val similarItems = clusterMemberships(cluster)
                val filteredUserRated = userRated.filter(x => similarItems.contains(x._1)).map(x => (x._1, x._2, jaccard(encoded{labels(itemToPredict)}, encoded{labels(x._1)})))
                val chosenNeighbors = math.min(filteredUserRated.size, itemNeighborThreshold)
                val userItemSimilarity = 5 * cosine(userProfile(user), encoded{labels{itemToPredict}}.map(x => x.toDouble))
                if(chosenNeighbors > 0){
                    val rating = filteredUserRated.toList.sortBy(-_._3).take(chosenNeighbors).map(x => x._2).sum / chosenNeighbors
                    val weightedRating = weightage * rating + userItemSimilarity * (1 - weightage)
                    weightedRating
                }else{
                    weightage * userHistorySummed(user) / userHistory(user).size + userItemSimilarity * (1 - weightage)
                }
            }
            else{
                userHistorySummed(user) / userHistory(user).size
            }
        } else if(!userHistory.contains(user))
            standardRatingDistrubution()
        else {

            // Using collaborative similarity index
            val userRated = userHistory(user)
            val userAverage = userHistorySummed(user) / userRated.size

            val similarities = userCFPearson(user)
            val neighborhood = math.min(userNeighborThreshold, similarities.size)
            val ratedUsers = itemHistory(itemToPredict)
            val consideredNeighbours = similarities.toList.filter(x => ratedUsers.contains(x._1)).sortBy(x => -math.abs(x._2)).take(neighborhood)
            val norm = consideredNeighbours.map(x => math.abs(x._2)).sum

            var summer = 0.0
            for (neighbour <- consideredNeighbours) {
                val neighbourAverage = (userHistorySummed(neighbour._1) - userHistory(neighbour._1)(itemToPredict)) / (userHistory.get(neighbour._1).size - 1)
                if(userHistory.get(neighbour._1).size - 1 != 0)
                    summer += (ratedUsers(neighbour._1) - neighbourAverage) * similarities(neighbour._1)
            }

            if(norm == 0.0)
                return userAverage
            val userCF = userAverage + summer/norm
            userCF




        }
    }

    /**
      * Calculates the correlation between two users.
      * @param corated the set of corated items with ratings from two users
      * @return pearson correlation between two users.
      */

    def pearson(corated: Set[(Int, (Double, Double))]): Double = {

        var item1Average = 0.0
        var item2Average = 0.0
        for(coratedItem <- corated){
            item1Average += coratedItem._2._1 / corated.size
            item2Average += coratedItem._2._2 / corated.size
        }
        var item1ratingSum = 0.0
        var item2ratingSum = 0.0
        var nr = 0.0
        for(coratedItem <- corated){
            nr = (coratedItem._2._1 - item1Average) * (coratedItem._2._2 - item2Average)
            item1ratingSum += math.pow(coratedItem._2._1 - item1Average,2)
            item2ratingSum += math.pow(coratedItem._2._2 - item2Average,2)
        }
        val dr = math.sqrt(item1ratingSum)*math.sqrt(item2ratingSum)

        if(dr == 0)
            return 0.0
        nr/dr
    }

    def getUserFeatures(userHistory: scala.collection.Map[Int, scala.collection.Map[Int, Double]], encoded: List[Array[Int]], labels: Map[Int, Int]): Map[Int, Array[Double]] ={
        val userFeatures = userHistory.map{
            x =>
                (x._1 , x._2.filter(y => labels.contains(y._1)).map(y => encoded{labels{y._1}}).reduce((y,z) => Utilities.add(y,z)))
        }
        userFeatures.map(x => (x._1, x._2.map(y => y.toDouble/userHistory(x._1).size))).toMap
    }

}
