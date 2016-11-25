
import scala.math.random
import scala.collection.mutable

import org.apache.spark._
import org.apache.spark.SparkContext._

val Tn = Array(Array(0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.1),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1))

/** 0-based index
Tn(2)(2) = 0.8
Tn(1)(1) = 0.8
Tn(1)(3) = 0.0
Tn(0)(0) = 0.9
Tn(2) = Array(0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
*/


val Ts =Array(Array(0.1, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9))

val Te =Array(Array(0.1, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.0, 0.0, 0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
            Array(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.8),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9))

val Tw =Array(Array(0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
            Array(0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0),
            Array(0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111),
            Array(0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0),
            Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1))



val n_s = 11
val n_a = 4

val T = Array.fill(n_s, n_a, n_s)(0.0)
// T[s,a,sp]: s current state; a action; sp next state
val R = Array.fill(n_s,n_a,n_s)(0.0)

var Q = Array.fill(n_s,n_a)(0.0)

val gamma:Double = 0.95
var eps_explore:Double = 0.2 // ^ 0.9^t
var alpha:Double = 0.1
var test1:String = ""


// 0-based index
// for-loop iteration includes start AND end index
for (i <- 0 to (n_s-1)) {
    for (j <- 0 to (n_s-1)) {
        T(i)(0)(j) = Tn(i)(j)
        T(i)(1)(j) = Ts(i)(j)
        T(i)(2)(j) = Te(i)(j)
        T(i)(3)(j) = Tw(i)(j)
    }
}


for (j <-  0 to (n_s-1)) {
    for (a <- 0 to (n_a-1)) {
        R(0)(a)(j) = -0.04
        R(1)(a)(j) = -0.04
        R(2)(a)(j) = -0.04
        R(3)(a)(j) = 1.0
        R(4)(a)(j) = -0.04
        R(5)(a)(j) = -0.04
        R(6)(a)(j) = -1.0
        R(7)(a)(j) = -0.04
        R(8)(a)(j) = -0.04
        R(9)(a)(j) = -0.04
        R(10)(a)(j) = -0.04
    }
}




def eps_greedy(q_vector:Array[Double], eps_explore:Double) : Int = {
    // q_vector is the row of Q function with current state s, i.e., q_vector =  Q[s,:]
    // eps_explore is the probability to explore, not to choose the best action

    var n_a = q_vector.length // number of actions

    var eps_equivalent:Double = eps_explore * n_a / (n_a - 1) // for convenience of sampling

    var best_action:Int = q_vector.indexOf(q_vector.max)

    var x:Double = scala.util.Random.nextDouble()

    if (x < eps_equivalent) {
        // to explore
        return scala.util.Random.nextInt(n_a) // include 0 and n_a-1
    }
    else
    {
        // to exploit the current best action
        return best_action
    }
}

def sample_in_machine(Trans:Array[Array[Array[Double]]], Rwd:Array[Array[Array[Double]]], s:Int, a:Int) 
        : Tuple2[Int, Double] = 
{
    // Given the transition function T and reward function R
    // with input s and a
    // sampling the next state sp and the reward r

    var n_s = T.length
    var distr = T(s)(a)

    var cdf_distr = Array.fill(n_s+1)(0.0)

    cdf_distr(0) = 0.0

    for (i <- 1 to n_s) {
        cdf_distr(i) = cdf_distr(i-1) + distr(i-1)
    }

    cdf_distr(n_s) = 1.0

    var x:Double = scala.util.Random.nextDouble()
    var sp:Int = 0

    for (i <- 0 to (n_s-1)) {
        if ((cdf_distr(i+1) > x) && (cdf_distr(i) <= x)) {
            sp = i
        }
    }

    var r = R(s)(a)(sp)


    return (sp,r)
}

def mapper(s:Int) : Tuple2[Tuple2[Int, Int], Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]]] = {

    // control
    var a:Int = eps_greedy(Q(s), eps_explore)
    // sample
    var pair_sp_r = sample_in_machine(T,R,s,a)
    // update Q table
    var qval = Q(s)(a) + alpha * (pair_sp_r._2 + gamma * Q(pair_sp_r._1).max - Q(s)(a))
    // why does only alpha-part not work, why include (1-alpha) here?
    var map = scala.collection.mutable.Map.empty[Int, Int]
    map += (pair_sp_r._1 -> 1) // a hash table (histogram over state space), can be accumulated (reduced)
    return ((s, a), ((qval, 1), map)) // (qval, 1) is also for reducer, sum and count -> get avg
}

def JointQReducer(
  A:Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]], 
  B:Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]])
  : Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]] = {
    var ret:Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]]
      = ((A._1._1 + B._1._1, A._1._2 + B._1._2), scala.collection.mutable.Map.empty[Int, Int])
    for ((k,v) <- A._2)
    {
        if (ret._2.contains(k)) ret._2(k) = ret._2(k) + v
        else ret._2 += (k -> v)
    }
    for ((k,v) <- B._2)
    {
        if (ret._2.contains(k)) ret._2(k) = ret._2(k) + v
        else ret._2 += (k -> v)
    }
    return ret
}

def ArbitQReducer(
  A:Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]], 
  B:Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]])
  : Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]] = {
    var ret:Tuple2[Tuple2[Double, Int], scala.collection.mutable.Map[Int, Int]]
      = ((A._1._1, A._1._2), scala.collection.mutable.Map.empty[Int, Int])
    // only keep A._1, drop B._1
    for ((k,v) <- A._2)
    {
        if (ret._2.contains(k)) ret._2(k) = ret._2(k) + v
        else ret._2 += (k -> v)
    }
    for ((k,v) <- B._2)
    {
        if (ret._2.contains(k)) ret._2(k) = ret._2(k) + v
        else ret._2 += (k -> v)
    }
    return ret
}

def updateQ(redResList:Array[
                        Tuple2[
                            Tuple2[Int, Int], 
                            Tuple2[
                                Tuple2[Double, Int], 
                                scala.collection.mutable.Map[Int, Int]]
                              ]
                            ], 
            p:Int) : Array[Int] = 
{
    var cnt:Int = 0
    var pS = Array.fill(p)(0)

    for (i <- 0 to redResList.length-1)
    {
        /*
        var coeff:Double = scala.math.pow(1-alpha, redResList(i)._2._1._2)
        Q(redResList(i)._1._1)(redResList(i)._1._2) = 
          coeff*Q(redResList(i)._1._1)(redResList(i)._1._2) +
          (1.0-coeff)*(redResList(i)._2._1._1/redResList(i)._2._1._2)
        */
        Q(redResList(i)._1._1)(redResList(i)._1._2) = redResList(i)._2._1._1/redResList(i)._2._1._2

        for ((k, v) <- redResList(i)._2._2)
        {
            for(j <- 1 to v)
            {
                pS(cnt) = k
                cnt += 1
            }
        }
    }
    if (cnt != p) println(s"Error: cnt = $cnt, but p = $p")
    return pS
}

def Q_learning_4x3_new(s:Int, p:Int, t_max:Int, mode:String) : Array[Array[Double]] = {

    // s is the initial state
    // t_max is the max number of iteration
    // gamma is the discount factor

    var t = 0

    var pS = Array.fill(p)(s) // same initial state 

    while (t < t_max) 
    {
        val distPS = sc.parallelize(pS) // partition by default
        val mapRes = distPS.map(mapper)

        if (mode == "Joint")
        {
            val redRes = mapRes.reduceByKey(JointQReducer).collect()
            pS = updateQ(redRes, p)
            test1 = mode
        }
        else if (mode == "Arbitrary")
        {
            val redRes = mapRes.reduceByKey(ArbitQReducer).collect()
            pS = updateQ(redRes, p)
            test1 = mode
        }

        t = t + 1
        if (t % 100 == 0) println(s"$t")
    }

    return Q
}

var res = Q_learning_4x3_new(0, 100, 300, "Arbitrary") //0-based index
test1
println("reset Q for next round")
Q = Array.fill(n_s,n_a)(0.0)

/**
How to debug the code?
1. print out some information to help you find the problem (like binary search)
2. if you have interactive command window, SUPERRRR! 
   You can just run line by line to approach the problem

Verified by Spark Databricks:

var res = Q_learning_4x3_new(0, 100, 300, "Arbitrary") //0-based index
res: Array[Array[Double]] = Array(Array(1.7570852674534607, 1.5989468594225216, 1.9341437821926402, 1.6708273824555646), Array(1.9777083306119825, 1.9945905518273128, 2.1822880460047944, 1.7828629049282674), Array(2.1992562127268305, 1.7917903817751515, 2.4235286213786638, 2.0024767987917653), Array(2.60635903207071, 2.6350194008004744, 2.644150866463204, 2.667703983377483), Array(1.6423057539027903, 1.1798225239836624, 1.4096418199412544, 1.4109046551067856), Array(1.8998221899552825, 1.2599839421610473, 0.9836632587017844, 1.6973151064120264), Array(0.4831899561123364, 0.5168163166277482, 0.6389787232566821, 0.5440399516165692), Array(1.284258673865986, 1.076365460841027, 1.3638219254321955, 1.144585830613829), Array(1.3221356528930581, 1.341862012469069, 1.468810013701929, 1.2661024003422472), Array(1.6079388440691582, 1.4810417347580134, 1.303631220198425, 1.4298486662956822), Array(0.6885310692903206, 1.2660799576145754, 1.1051288037882943, 1.3786371484497637))
Command took 14.02s -- by yusheng@stanford.edu at 6/6/2016, 12:24:09 PM on My Cluster (6 GB)


var res = Q_learning_4x3_new(0, 100, 300, "Joint") //0-based index
test1
println("reset Q for next round")
Q = Array.fill(n_s,n_a)(0.0)

res: Array[Array[Double]] = Array(Array(1.8100772697163363, 1.6475972662926148, 1.990046318781084, 1.7609951400714483), Array(2.072311903279266, 2.038066199669534, 2.2427476638058517, 1.8578388383526856), Array(2.286933263693919, 1.9742354095433563, 2.5043557874913067, 2.090148438460775), Array(2.6907786421413222, 2.794040162972387, 2.7390553003317826, 2.8072823565510507), Array(1.7748268642170333, 1.317603941819252, 1.4333005881717562, 1.5675660132383118), Array(2.102176156656342, 1.4827123188643192, 1.019998242914896, 1.9378229752128076), Array(0.7706136568828748, 0.4145357587153916, 0.46783785375795284, 0.533776234667545), Array(1.3719127422219308, 1.1454352366986436, 1.478743949409724, 1.1424311543194638), Array(1.4185142561281472, 1.447600396877467, 1.6422118366971208, 1.2875552018212535), Array(1.8499596991152771, 1.600543910126896, 1.415074540261772, 1.495497581521719), Array(0.6931667698567034, 1.2535907174931833, 1.2064630744731388, 1.5436115191640902))
Command took 13.16s -- by yusheng@stanford.edu at 6/6/2016, 12:26:45 PM on My Cluster (6 GB)
 */

