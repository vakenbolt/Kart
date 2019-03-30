package io.samuelagesilas

import kotlin.math.pow

/**
 * Main class for the [DecisionTreeClassifier] where [T] indicates the type associated with the Classification column
 * in the training model.
 * @param trainingModel: The [List] of[TrainingModeledRow]'s making up the training model used to build the decision
 * tree.
 * @param predicates: The [List] of [Predicate]'s used to query the training model.
 */
class DecisionTreeClassifier<T>(private val trainingModel: List<TrainingModeledRow<T>>,
                                private val predicates: List<Predicate<*>>) {

    /**
     * The calculated Gini impurity for the given [trainingModel].
     */
    val rootGiniImpurity: Double = this.calculateGiniImpurity(trainingModel)

    /**
     * [Iterator] associated to the list of predicates sorted in descending order by information gain.
     */
    val sortedPredicates: Iterator<InformationGain<T>> = this.calculateInformationGain(rows = trainingModel)
            .sortedByDescending { it.informationGain }.iterator()

    private fun calculateGiniImpurity(trainingModelRows: List<TrainingModeledRow<T>>): Double {
        val classificationCounts: MutableMap<T, Int> = mutableMapOf()
        val distinctClasses: List<TrainingModeledRow<T>> = trainingModelRows.distinctBy { it.classification() }
        distinctClasses.forEach { trainingModelRow: TrainingModeledRow<T> ->
            classificationCounts[trainingModelRow.classification()] = trainingModelRows
                    .count { m: TrainingModeledRow<T> ->
                        m.classification()!!.toString() == trainingModelRow.classification().toString()
                    }
        }
        val probabilities: MutableList<Double> = mutableListOf()
        classificationCounts.forEach { _, v ->
            probabilities.add(v.toDouble().div(trainingModelRows.size).pow(2))
        }
        return 1 - probabilities.sum()
    }

    private fun evaluatePredicate(p: Predicate<TrainingModeledRow<T>>): PredicateResult<T> {
        val resolvedAsTrue: MutableList<TrainingModeledRow<T>> = mutableListOf()
        val resolvedAsFalse: MutableList<TrainingModeledRow<T>> = mutableListOf()
        trainingModel.iterator().forEach { row: TrainingModeledRow<T> ->
            when (p.invoke(row)) {
                true -> resolvedAsTrue.add(row)
                false -> resolvedAsFalse.add(row)
            }
        }
        return PredicateResult(left = resolvedAsTrue.toList(), right = resolvedAsFalse.toList())
    }

    private fun calculateInformationGain(rows: List<TrainingModeledRow<T>>): List<InformationGain<T>> {
        val predicateInformationGain: MutableList<InformationGain<T>> = mutableListOf()
        @Suppress("UNCHECKED_CAST")
        val p = predicates as List<Predicate<TrainingModeledRow<T>>>
        p.iterator().forEach { predicate: Predicate<TrainingModeledRow<T>> ->
            val result: PredicateResult<T> = evaluatePredicate(predicate)
            val leftGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.left)
            val rightGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.right)

            //weighted average of each question
            val avgImpurity: Double = (result.left.size.toDouble() / rows.size) * leftGiniImpurity
                    .plus(result.right.size.toDouble() / rows.size) * rightGiniImpurity
            val informationGain : Double = this.rootGiniImpurity - avgImpurity
            predicateInformationGain.add(InformationGain(predicate, avgImpurity, informationGain))
        }
        return predicateInformationGain.toList()

    }

    fun evaluate(row: TrainingModeledRow<T>): T {
        TODO("not implemented")
    }

    fun evaluate(rows: List<TrainingModeledRow<T>>): List<T> {
        TODO("not implemented")
    }
}


/**
 * Data class indicating the [predicate] associated with the [avgImpurity] and [informationGain].
 */
data class InformationGain<Classification>(val predicate: Predicate<TrainingModeledRow<Classification>>,
                                           val avgImpurity: Double,
                                           val informationGain: Double)
