package io.samuelagesilas

import kotlin.math.pow

/**
 * Main class for the [DecisionTreeClassifier] where [T] indicates the type associated with the classification column
 * in the training model.
 * @param trainingModel: The [List] of[TrainingModeledRow]'s making up the training model used to build the decision
 * tree.
 * @param predicateFunctions: The [List] of [PredicateFunction]'s used to query the training model.
 */
class DecisionTreeClassifier<T>(private val trainingModel: List<TrainingModeledRow<T>>,
                                private val predicateFunctions: List<PredicateFunction<*>>) {

    /**
     * The calculated Gini impurity for the given [trainingModel].
     */
    val rootGiniImpurity: Double = this.calculateGiniImpurity(trainingModel)

    /**
     * [List] associated to the list of predicate functions sorted in descending order by information gain.
     */
    val sortedPredicates: List<PredicateNode<T>> = this.calculateInformationGain(rows = trainingModel)
            .sortedByDescending { it.predicate.informationGain }

    private fun calculateGiniImpurity(trainingModelRows: List<TrainingModeledRow<T>>): Double {
        val classificationCounts: MutableMap<T, Int> = mutableMapOf()
        val distinctClasses: List<TrainingModeledRow<T>> = trainingModelRows.distinctBy { it.classification() }
        distinctClasses.forEach { trainingModelRow: TrainingModeledRow<T> ->
            classificationCounts[trainingModelRow.classification()] = trainingModelRows.count { m: TrainingModeledRow<T> ->
                m.classification()!!.toString() == trainingModelRow.classification().toString()
            }
        }
        val probabilities: MutableList<Double> = mutableListOf()
        classificationCounts.forEach { _, v ->
            probabilities.add(v.toDouble().div(trainingModelRows.size).pow(2))
        }
        return 1 - probabilities.sum()
    }

    private fun evaluatePredicate(p: PredicateFunction<TrainingModeledRow<T>>,
                                  trainingModel: List<TrainingModeledRow<T>>): PredicateResult<T> {
        val resolvedAsTrue: MutableList<TrainingModeledRow<T>> = mutableListOf()
        val resolvedAsFalse: MutableList<TrainingModeledRow<T>> = mutableListOf()
        trainingModel.iterator().forEach { row: TrainingModeledRow<T> ->
            when (p.function.invoke(row)) {
                true -> resolvedAsTrue.add(row)
                false -> resolvedAsFalse.add(row)
            }
        }
        return PredicateResult(left = resolvedAsTrue.toList(), right = resolvedAsFalse.toList())
    }

    private fun calculateInformationGain(rows: List<TrainingModeledRow<T>>): List<PredicateNode<T>> {
        val predicateInformationGain: MutableList<PredicateNode<T>> = mutableListOf()
        @Suppress("UNCHECKED_CAST")
        val p = predicateFunctions as List<PredicateFunction<TrainingModeledRow<T>>>
        p.iterator().forEach { predicateFunction: PredicateFunction<TrainingModeledRow<T>> ->
            val result: PredicateResult<T> = evaluatePredicate(predicateFunction, this.trainingModel)
            val leftGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.left)
            val rightGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.right)

            //weighted average of each question
            val avgImpurity: Double = (result.left.size.toDouble() / rows.size) * leftGiniImpurity
                    .plus(result.right.size.toDouble() / rows.size) * rightGiniImpurity
            val informationGain: Double = this.rootGiniImpurity - avgImpurity
            predicateInformationGain.add(PredicateNode(Predicate(predicateFunction, avgImpurity, informationGain)))
        }
        return predicateInformationGain.toList()
    }

    private tailrec fun processNode(currentNode: PredicateNode<T>,
                                    trainingModel: List<TrainingModeledRow<T>>,
                                    sortedPredicates: Iterator<PredicateNode<T>>) {
        val result: PredicateResult<T> = this.evaluatePredicate(currentNode.predicate.predicateFunction, trainingModel)

        //add right sight of to queue
        when {
            (result.left.size == 1) -> {
                //node is a leaf
                return
            }
            (result.right.size == 1) -> {
                //node is a leaf
                return
            }
            (result.left.size > 1) -> {
                //process next node
                when (sortedPredicates.hasNext()) {
                    true -> {
                        val nextNode: PredicateNode<T> = sortedPredicates.next()
                        currentNode.leftNode = nextNode
                        processNode(nextNode, result.left, sortedPredicates)
                    }
                    false -> {
                        //node is unresolved perform a guess
                    }
                }
            }
        }
    }

    init {
        this.sortedPredicates.forEach { println("${it.predicate.predicateFunction.label}, ${it.predicate.avgImpurity}, ${it.predicate.informationGain}") }
    }

    fun evaluate(row: TrainingModeledRow<T>): T {
        TODO("not implemented")
    }

    fun evaluate(rows: List<TrainingModeledRow<T>>): List<T> {
        TODO("not implemented")
    }
}
