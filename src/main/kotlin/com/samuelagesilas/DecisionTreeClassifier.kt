package com.samuelagesilas

import java.util.*
import kotlin.math.pow

/**
 * Main class for the [DecisionTreeClassifier] where [T] indicates the type associated with the classification column
 * in the training model.
 * @param trainingModel: The [List] of[DecisionTreeClassifierDataRow]'s making up the training model used to build the decision
 * tree.
 * @param predicateFunctions: The [List] of [PredicateFunction]'s used to query the training model.
 */
class DecisionTreeClassifier<T>(
        internal val trainingModel: List<DecisionTreeClassifierDataRow<T>>,
        private val predicateFunctions: List<PredicateFunction<*>>) {

    /**
     * The calculated Gini impurity for the given [trainingModel].
     */
    val rootGiniImpurity: Double = this.calculateGiniImpurity(trainingModel)

    /**
     * [List] associated to the list of predicate functions sorted in descending order by information gain.
     */
    val sortedPredicates: List<Predicate<T>> = this.calculateInformationGain(rows = trainingModel)
            .sortedByDescending { it.informationGain }

    private var decisionTree: PredicateNode<T>

    private fun calculateGiniImpurity(trainingModelRows: List<DecisionTreeClassifierDataRow<T>>): Double {
        val classificationCounts: MutableMap<T, Int> = mutableMapOf()
        val distinctClasses: List<DecisionTreeClassifierDataRow<T>> = trainingModelRows.distinctBy { it.classification() }
        distinctClasses.forEach { trainingModelRow: DecisionTreeClassifierDataRow<T> ->
            classificationCounts[trainingModelRow.classification()!!] = trainingModelRows.count { m: DecisionTreeClassifierDataRow<T> ->
                m.classification()!!.toString() == trainingModelRow.classification().toString()
            }
        }
        val probabilities: MutableList<Double> = mutableListOf()
        classificationCounts.forEach { _, v ->
            probabilities.add(v.toDouble().div(trainingModelRows.size).pow(2))
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
            val result: PredicateResult<T> = evaluatePredicate(predicateFunction, this.trainingModel)
            val leftGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.left)
            val rightGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.right)

            //weighted average of each question
            val avgImpurity: Double = (result.left.size.toDouble() / rows.size) * leftGiniImpurity
                    .plus(result.right.size.toDouble() / rows.size) * rightGiniImpurity
            val informationGain: Double = this.rootGiniImpurity - avgImpurity
            predicateInformationGain.add(Predicate(predicateFunction, avgImpurity, informationGain))
        }
        return predicateInformationGain.toList()
    }

    private fun classify(node: PredicateNode<T>,
                         classifications: List<DecisionTreeClassifierDataRow<T>>): DecisionTreeClassifierDataRow<T> = when {
        (classifications.size == 1) -> classifications.first()
        (classifications.size > 1) -> classifications[(0..(classifications.size - 1)).random()]
        else -> classifications[(0..(node.nodeResult!!.size - 1)).random()]
    }

    private tailrec fun evaluateWithTree(row: DecisionTreeClassifierDataRow<T>,
                                         node: PredicateNode<T>): T {
        with(node.nodeResult!!) {
            if (this.size == 1 || this.isEmpty()) {
                return classify(node, classifications = this).classification()!!
            }
        }
        val result: Boolean = node.predicate.predicateFunction.function.invoke(row)
        val nextNode: PredicateNode<T> = if (result) {
            if (node.leftNode == null) {
                return classify(node, classifications = node.nodeResult!!).classification()!!
            }
            node.leftNode!!
        } else {
            if (node.rightNode == null) {
                return classify(node, classifications = node.nodeResult!!).classification()!!
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

    private fun processNode(rootNode: PredicateNode<T>,
                            childNodes: LinkedList<PredicateNode<T>>,
                            evaluatePredicate: (p: PredicateFunction<DecisionTreeClassifierDataRow<T>>,
                                                trainingModel: List<DecisionTreeClassifierDataRow<T>>) -> PredicateResult<T>) {
        if (rootNode.nodeResult!!.size == 1) return
        rootNode.result = evaluatePredicate.invoke(rootNode.predicate.predicateFunction,
                                                   rootNode.nodeResult!!)
        val result: PredicateResult<T> = rootNode.result!!
        val predicateIndex: Int = rootNode.predicateIndex!!
        if (result.left.isNotEmpty() && predicateIndex < this.sortedPredicates.lastIndex) {
            val index: Int = predicateIndex + 1
            rootNode.leftNode = PredicateNode(this.sortedPredicates[index], index, result.left)
            childNodes.push(rootNode.leftNode)
        }
        if (result.right.isNotEmpty() && predicateIndex < this.sortedPredicates.lastIndex) {
            val index: Int = predicateIndex + 1
            rootNode.rightNode = PredicateNode(this.sortedPredicates[index], index, result.right)
            childNodes.push(rootNode.rightNode)
        }
    }

    /**
     * Returns the root node of the generated decision tree classifier. The resulting tree is used by the
     * [DecisionTreeClassifier] to evaluate data.
     */
    internal fun buildDecisionTree(): PredicateNode<T> {
        val childNodes: LinkedList<PredicateNode<T>> = LinkedList()
        val rootNode: PredicateNode<T> = PredicateNode(predicate = this.sortedPredicates.first(),
                                                       predicateIndex = 0,
                                                       nodeResult = this.decisionTreeClassifier.trainingModel)
        processNode(rootNode, childNodes, decisionTreeClassifier::evaluatePredicate)
        while (childNodes.size > 0) {
            processNode(childNodes.poll(), childNodes, decisionTreeClassifier::evaluatePredicate)
        }
        return rootNode
    }
}