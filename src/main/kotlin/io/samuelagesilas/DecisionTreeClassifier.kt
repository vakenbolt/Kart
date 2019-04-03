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

    /**
     * The calculated Gini impurity for the given [trainingData].
     */
    val rootGiniImpurity: Double = this.calculateGiniImpurity(trainingData)

    /**
     * [List] associated to the list of predicate functions sorted in descending order by information gain.
     */
    val sortedPredicates: List<Predicate<T>> = this.calculateInformationGain(rows = trainingData)
        .sortedByDescending { it.informationGain }
        .let { p: List<Predicate<T>> ->
            val filteredList: MutableList<Predicate<T>> = mutableListOf(p[0])
            for (i in 1 until p.indexOfFirst { i: Predicate<T> -> i.informationGain == 0.0 }) {
                if (p[i] != p[0] && p[i].informationGain != p[0].informationGain) filteredList.add(p[i])
            }
            filteredList
        }

    private var decisionTree: PredicateNode<T>

    private fun calculateGiniImpurity(trainingDataRows: List<DecisionTreeClassifierDataRow<T>>): Double {
        val classificationCounts: MutableMap<T, Int> = mutableMapOf()
        val distinctClassification: List<DecisionTreeClassifierDataRow<T>> = trainingDataRows.distinctBy {
            it.classification()
        }
        //check for the number of times a classification appears in the trainingDataRows list.
        distinctClassification.forEach { trainingDataRow: DecisionTreeClassifierDataRow<T> ->
            classificationCounts[trainingDataRow.classification()!!] =
                trainingDataRows.count { m: DecisionTreeClassifierDataRow<T> ->
                    m.classification()!!.toString() == trainingDataRow.classification().toString()
                }
        }
        val probabilities: MutableList<Double> = mutableListOf()
        classificationCounts.forEach { _, v ->
            probabilities.add((v.toDouble() / trainingDataRows.size).pow(2))
        }
        return 1 - probabilities.sum()
    }

    /**
     * Performs a partition on the current [filteredTrainingData] using the provided [PredicateFunction] [pFunc].
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
        return PredicateResult(left = resolvedAsTrue.toList(), right = resolvedAsFalse.toList())
    }

    private fun calculateInformationGain(rows: List<DecisionTreeClassifierDataRow<T>>): List<Predicate<T>> {
        val predicateInformationGain: MutableList<Predicate<T>> = mutableListOf()
        @Suppress("UNCHECKED_CAST")
        val p = predicateFunctions as List<PredicateFunction<DecisionTreeClassifierDataRow<T>>>
        p.iterator().forEach { predicateFunction: PredicateFunction<DecisionTreeClassifierDataRow<T>> ->
            val result: PredicateResult<T> = evaluatePredicate(predicateFunction, this.trainingData)
            val leftGiniImpurity: Double = this.calculateGiniImpurity(trainingDataRows = result.left)
            val rightGiniImpurity: Double = this.calculateGiniImpurity(trainingDataRows = result.right)

            //weighted average of each question
            val avgImpurity: Double = (result.left.size.toDouble() / rows.size) * leftGiniImpurity +
                    (result.right.size.toDouble() / rows.size) * rightGiniImpurity
            val informationGain: Double = this.rootGiniImpurity - avgImpurity
            predicateInformationGain.add(Predicate(predicateFunction, avgImpurity, informationGain))
        }
        return predicateInformationGain.toList()
    }

    private fun classify(node: PredicateNode<T>,
                         partition: List<DecisionTreeClassifierDataRow<T>>): DecisionTreeClassifierDataRow<T> {
        return when {
            (partition.size == 1) -> partition.first()
            (partition.size > 1) -> partition[(0..(partition.size - 1)).random()]
            else -> partition[(0..(node.nodeResult!!.size - 1)).random()]
        }
    }


    private tailrec fun evaluateWithTree(row: DecisionTreeClassifierDataRow<T>,
                                         node: PredicateNode<T>): T {
        with(node.nodeResult!!) {
            if (this.size == 1
                || this.isEmpty()
                || node.predicateFunction == null
            ) return classify(node, partition = this).classification()!!
        }
        val result: Boolean = node.predicateFunction!!.function.invoke(row)
        val nextNode: PredicateNode<T> = if (result) {
            if (node.leftNode == null) {
                return classify(node, partition = node.nodeResult).classification()!!
            }
            node.leftNode!!
        } else {
            if (node.rightNode == null) {
                return classify(node, partition = node.nodeResult).classification()!!
            }
            node.rightNode!!
        }
        return evaluateWithTree(row, nextNode)
    }


    init {
        val f = DecisionTreeNodeBuilder<T>(decisionTreeClassifier = this)
        this.decisionTree = f.buildDecisionTree()
    }

    fun evaluate(row: DecisionTreeClassifierDataRow<T>): T = evaluateWithTree(row, decisionTree)

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
    private val sortedPredicates: List<Predicate<T>> = decisionTreeClassifier.sortedPredicates

    private tailrec fun processNode(rootNode: PredicateNode<T>,
                                    iterator: Iterator<Predicate<T>>,
                                    rightNodes: LinkedList<PredicateNode<T>>,
                                    evaluatePredicate: (p: PredicateFunction<DecisionTreeClassifierDataRow<T>>,
                                                        trainingData: List<DecisionTreeClassifierDataRow<T>>) -> PredicateResult<T>) {

        if (!iterator.hasNext()) return
        with(rootNode.nodeResult!!) { if (this.size == 1 || this.map { it.classification() }.distinct().size == 1) return }
        val f: PredicateFunction<DecisionTreeClassifierDataRow<T>> = iterator.next().predicateFunction
        rootNode.result = evaluatePredicate.invoke(f, rootNode.nodeResult)
        rootNode.predicateFunction = f

        val result: PredicateResult<T> = rootNode.result!!
        if (result.left.isNotEmpty() || result.left.size == 1 || result.left.map { it.classification() }.distinct().size == 1) {
            rootNode.leftNode = PredicateNode(result.left)
            rootNode.rightNode = PredicateNode(result.right)
            rightNodes.push(rootNode.rightNode)
            processNode(rootNode.leftNode!!, iterator, rightNodes, evaluatePredicate)
        }
    }

    /**
     * Returns the root node of the generated decision tree classifier. The resulting tree is used by the
     * [DecisionTreeClassifier] to evaluate data.
     */
    internal fun buildDecisionTree(): PredicateNode<T> {
        val rightNodes: LinkedList<PredicateNode<T>> = LinkedList()
        val rootNode: PredicateNode<T> = PredicateNode(nodeResult = this.decisionTreeClassifier.trainingData)
        val sortedPredicatesIterator: Iterator<Predicate<T>> = this.sortedPredicates.iterator()
        processNode(rootNode, sortedPredicatesIterator, rightNodes, decisionTreeClassifier::evaluatePredicate)
        while (rightNodes.size > 0) {
            processNode(rightNodes.poll(),
                        sortedPredicatesIterator,
                        rightNodes,
                        decisionTreeClassifier::evaluatePredicate)
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