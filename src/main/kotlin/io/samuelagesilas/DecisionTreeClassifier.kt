package io.samuelagesilas

import java.util.*
import kotlin.math.pow

/**
 * Main class for the [DecisionTreeClassifier] where [T] indicates the type associated with the classification column
 * in the training data.
 * @param trainingData: The [List] of[DecisionTreeClassifierDataRow]'s making up the training data used to build the
 * decision tree.
 * @param predicateFunctions: The [List] of [PredicateFunction]'s used to query the training data.
 */
class DecisionTreeClassifier<T>(internal val trainingData: List<DecisionTreeClassifierDataRow<T>>,
                                private val predicateFunctions: List<PredicateFunction<*>>) {

    private var decisionTree: PredicateNode<T> = DecisionTreeNodeBuilder<T>(decisionTreeClassifier = this).build()

    internal fun calculateGiniImpurity(rows: List<DecisionTreeClassifierDataRow<T>>): Double {
        val classificationCounts: MutableMap<T, Int> = mutableMapOf()
        val distinctClassification: List<DecisionTreeClassifierDataRow<T>> = rows.distinctBy {
            it.classification()
        }
        //check for the number of times a classification appears in the trainingDataRows list.
        distinctClassification.forEach { trainingDataRow: DecisionTreeClassifierDataRow<T> ->
            classificationCounts[trainingDataRow.classification()!!] =
                rows.count { m: DecisionTreeClassifierDataRow<T> ->
                    m.classification()!!.toString() == trainingDataRow.classification().toString()
                }
        }
        //Gini impurity is calculated by the sum of the probabilities of a particularly classification.
        val probabilities: MutableList<Double> = mutableListOf()
        classificationCounts.forEach { _, v ->
            probabilities.add((v.toDouble() / rows.size).pow(2))
        }
        return 1 - probabilities.sum()
    }

    /**
     * Performs a partition on the current [filteredTrainingData] using the provided [PredicateFunction] [pFunc]. The
     * left side fill always contain the partition data for a predicate that returns false, opposite is true for
     * predicates that return true.
     */
    internal fun evaluatePredicate(pFunc: PredicateFunction<DecisionTreeClassifierDataRow<T>>,
                                   filteredTrainingData: List<DecisionTreeClassifierDataRow<T>>): PredicateResult<T> {
        val resolvedAsTrue: MutableList<DecisionTreeClassifierDataRow<T>> = mutableListOf()
        val resolvedAsFalse: MutableList<DecisionTreeClassifierDataRow<T>> = mutableListOf()
        filteredTrainingData.iterator().forEach { row: DecisionTreeClassifierDataRow<T> ->
            when (pFunc.function.invoke(row)) {
                true -> resolvedAsTrue.add(row)
                false -> resolvedAsFalse.add(row)
            }
        }
        return PredicateResult(left = resolvedAsFalse.toList(), right = resolvedAsTrue.toList())
    }

    internal fun calculateInformationGain(rows: List<DecisionTreeClassifierDataRow<T>>): List<Predicate<T>> {
        val predicateInformationGain: MutableList<Predicate<T>> = mutableListOf()
        val giniImpurityForAllRows : Double = this.calculateGiniImpurity(rows)
        //iterate through the list of predicates to find the one with the highest information gain
        @Suppress("UNCHECKED_CAST")
        val p = predicateFunctions as List<PredicateFunction<DecisionTreeClassifierDataRow<T>>>
        p.iterator().forEach { predicateFunction: PredicateFunction<DecisionTreeClassifierDataRow<T>> ->
            val result: PredicateResult<T> = evaluatePredicate(predicateFunction, rows)
            val leftGiniImpurity: Double = this.calculateGiniImpurity(rows = result.left)
            val rightGiniImpurity: Double = this.calculateGiniImpurity(rows = result.right)

            //weighted average of each question
            val avgImpurity: Double = (result.left.size.toDouble() / rows.size) * leftGiniImpurity +
                    (result.right.size.toDouble() / rows.size) * rightGiniImpurity
            val informationGain: Double = giniImpurityForAllRows - avgImpurity
            predicateInformationGain.add(Predicate(predicateFunction, avgImpurity, informationGain))
        }
        return predicateInformationGain.toList()
    }

    /**
     * Returns the classification of [node] associated with the given [partition].
     */
    private fun classify(node: PredicateNode<T>,
                         partition: List<DecisionTreeClassifierDataRow<T>>): T {
        return when {
            (partition.size == 1) -> partition.first()
            //if the partition size is greater than 1, we must randomly choose a classification from the partitioned list
            (partition.size > 1) -> partition[(0..(partition.size - 1)).random()]
            //if the partition is empty, randomly choose a classification from the result of predicate
            else -> partition[(0..(node.nodeResult!!.size - 1)).random()]
        }.classification()!!
    }


    private tailrec fun evaluateWithTree(row: DecisionTreeClassifierDataRow<T>,
                                         node: PredicateNode<T>): T {
        with(node.nodeResult!!) {
            if (this.size == 1
                || this.isEmpty()
                || node.predicateFunction == null
            ) return classify(node, partition = this)
        }
        val result: Boolean = node.predicateFunction!!.function.invoke(row)
        val nextNode: PredicateNode<T> = if (!result) {
            if (node.leftNode == null) {
                return classify(node, partition = node.nodeResult)
            }
            node.leftNode!!
        } else {
            if (node.rightNode == null) {
                return classify(node, partition = node.nodeResult)
            }
            node.rightNode!!
        }
        return evaluateWithTree(row, nextNode)
    }


    /**
     * Returns the classification of [row].
     */
    fun evaluate(row: DecisionTreeClassifierDataRow<T>): T = evaluateWithTree(row, decisionTree)

    /**
     * Takes a [List] of [rows] and returns a list of associated classifications.
     */
    fun evaluate(rows: List<DecisionTreeClassifierDataRow<T>>): List<T> {
        val l = mutableListOf<T>()
        rows.forEach { row -> l.add(evaluateWithTree(row, decisionTree)) }
        return l.toList()
    }
}


/**
 * Builds the decision tree using the provided [decisionTreeClassifier].
 */
class DecisionTreeNodeBuilder<T>(private val decisionTreeClassifier: DecisionTreeClassifier<T>) {

    private tailrec fun processNode(rootNode: PredicateNode<T>,
                                    rightNodes: LinkedList<PredicateNode<T>>,
                                    evaluatePredicate: (p: PredicateFunction<DecisionTreeClassifierDataRow<T>>,
                                                        trainingData: List<DecisionTreeClassifierDataRow<T>>) -> PredicateResult<T>) {

        with(rootNode.nodeResult!!) {
            if (this.size == 1 || this.map { it.classification() }.distinct().size == 1) return
        }

        //predicate with the highest information gain against the current filtered set
        val bestPredicate = this.decisionTreeClassifier
            .calculateInformationGain(rootNode.nodeResult)
            .sortedByDescending { it.informationGain }
            .first()
        if (bestPredicate.informationGain == 0.0) return

        val pFunc: PredicateFunction<DecisionTreeClassifierDataRow<T>> = bestPredicate.predicateFunction
        rootNode.result = evaluatePredicate.invoke(pFunc, rootNode.nodeResult)
        if (rootNode.result!!.left.isEmpty() || rootNode.result!!.right.isEmpty()) return
        rootNode.predicateFunction = pFunc

        val result: PredicateResult<T> = rootNode.result!!
        if (result.right.isNotEmpty() || result.right.size == 1 || result.left.map { it.classification() }.distinct().size == 1) {
            rootNode.leftNode = PredicateNode(result.left)
            rootNode.rightNode = PredicateNode(result.right)
            rightNodes.push(rootNode.leftNode)
            processNode(rootNode.rightNode!!, rightNodes, evaluatePredicate)
        }
    }

    /**
     * Returns the root node of the generated decision tree classifier. The resulting tree is used by the
     * [DecisionTreeClassifier] to evaluate data.
     */
    internal fun build(): PredicateNode<T> {
        val leftNodes: LinkedList<PredicateNode<T>> = LinkedList()
        val rootNode: PredicateNode<T> = PredicateNode(nodeResult = this.decisionTreeClassifier.trainingData)
        processNode(rootNode, leftNodes, decisionTreeClassifier::evaluatePredicate)
        while (leftNodes.size > 0) {
            processNode(leftNodes.poll(), leftNodes, decisionTreeClassifier::evaluatePredicate)
        }
        return rootNode
    }
}


/**
 * Interface for any data type used as a row in a training data.
 * [T] indicates the type associated with the classification column in the training data.
 */
abstract class DecisionTreeClassifierDataRow<T> {

    /**
     * Returns the typed value associated with the classification column in the training data.
     */
    open fun classification(): T? {
        return null
    }
}