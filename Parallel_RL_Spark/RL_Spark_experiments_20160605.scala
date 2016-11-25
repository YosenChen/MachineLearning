
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

def mapper(s:Int) : Tuple4[Int, Int, Double, Int] = {

    // control
    var a:Int = eps_greedy(Q(s), eps_explore)
    // sample
    var pair_sp_r = sample_in_machine(T,R,s,a)
    // update Q table
    var qval = Q(s)(a) + alpha * (pair_sp_r._2 + gamma * Q(pair_sp_r._1).max - Q(s)(a))
    return (s, a, qval, pair_sp_r._1)
}

def updateQ(resList:Array[Tuple4[Int, Int, Double, Int]], mode:String) = {

    if (mode == "Arbitrary")
    {
        for (i <- 0 to resList.length-1)
        {
            Q(resList(i)._1)(resList(i)._2) = resList(i)._3
        }
        test1 = mode
    }
    else if (mode == "Joint")
    {
        // construct hasp map
        var map = scala.collection.mutable.Map.empty[Tuple2[Int, Int], scala.collection.mutable.ArrayBuffer[Double]]
        for (i <- 0 to resList.length-1)
        {
            if (map.contains((resList(i)._1, resList(i)._2)))
            {
                map((resList(i)._1, resList(i)._2)) += resList(i)._3
            }
            else
            {
                map += ((resList(i)._1, resList(i)._2) -> scala.collection.mutable.ArrayBuffer(resList(i)._3))
            }
        }

        for ((k,v) <- map)
        {
            var coeff:Double = scala.math.pow(1-alpha, v.length)
            Q(k._1)(k._2) = coeff*Q(k._1)(k._2) + (1.0-coeff)*(v.sum/v.length)
        }
        test1 = mode
    }
}

def Q_learning_4x3_new(s:Int, p:Int, t_max:Int, mode:String) : Array[Array[Double]] = {

    // s is the initial state
    // t_max is the max number of iteration
    // gamma is the discount factor

    var t = 0

    val pS = Array.fill(p)(s) // same initial state 

    while (t < t_max) 
    {
        val distPS = sc.parallelize(pS) // partition by default
        val res = distPS.map(mapper).cache().collect()

        // update Q
        updateQ(res, mode)
        for (i <- 0 to (p-1))
        {
            pS(i) = res(i)._4
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
/*
res(0)
res(1)
res(2)
res(3)
res(4)
res(5)
res(6)
res(7)
res(8)
res(9)
res(10)
 */



/**
scala> :load ..\examples\src\main\scala\org\apache\spark\examples\YosenSelfExample\RL_Spark_experiments.scala
....
 */

/**
You can use :reset
to clear all variables in Scala REPL before each test
in Spark, you can restart cluster, but I suggest manually clear all variables, 
since restarting cluster takes very long


Verified at Spark Databricks
res: Array[Array[Double]] = Array(Array(1.8052091255281146, 1.6575821359275715, 2.028214304325208, 1.7454440349650775), Array(2.007542667952362, 2.0528286660485744, 2.275657135526577, 1.922924140342297), Array(2.2717938768454085, 2.017561572398659, 2.4812650368427844, 2.0967015991095193), Array(2.6111435555943303, 2.6287277550497374, 2.7894259427137156, 2.6237025207192355), Array(1.7802029523378966, 1.4183785845290937, 1.6222645938483387, 1.5532894468052678), Array(2.2659828306500915, 1.5447689929965076, 0.9499706662208331, 1.9668197560764438), Array(0.5216730750874967, 0.7364349815044954, 0.6307248952943576, 0.6300600522424149), Array(1.581895711689666, 1.2773774532446305, 1.348648874436817, 1.2301659922720063), Array(1.3386587934602534, 1.3339259045244864, 1.6442159595299521, 1.311942134932542), Array(1.9489982925789553, 1.5202551456818645, 1.4410851241541247, 1.4688858976965262), Array(0.8214812176129646, 1.2370570229456528, 1.0739003704124488, 1.5725710597141056))

Command took 7.08s -- by yusheng@stanford.edu at 6/6/2016, 11:52:45 AM on My Cluster (6 GB)
 */