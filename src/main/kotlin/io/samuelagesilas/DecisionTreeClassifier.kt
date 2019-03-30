package io.samuelagesilas

import kotlin.math.pow

class DecisionTreeClassifier<Classification>(private val trainingModel: List<TrainingModeledRow<Classification>>,
                                             private val predicates: Any) {

    val rootGiniImpurity: Double = this.calculateGiniImpurity(trainingModel)

    val sortedPredicates: Iterator<InformationGain<Classification>> = this.calculateInformationGain(rows = trainingModel)
            .sortedByDescending { it.avgImpurity }.iterator()

    init {
        //Use sortedPredicates to build decision tree
    }

    private fun calculateGiniImpurity(trainingModelRows: List<TrainingModeledRow<Classification>>): Double {
        val classificationCounts: MutableMap<Classification, Int> = mutableMapOf()
        val distinctClasses: List<TrainingModeledRow<Classification>> = trainingModelRows.distinctBy { it.classification() }
        distinctClasses.forEach { trainingModelRow: TrainingModeledRow<Classification> ->
            classificationCounts[trainingModelRow.classification()] = trainingModelRows
                    .count { m: TrainingModeledRow<Classification> ->
                        m.classification()!!.toString() == trainingModelRow.classification().toString()
                    }
        }
        val probabilities: MutableList<Double> = mutableListOf()
        classificationCounts.forEach { _, v ->
            probabilities.add(v.toDouble().div(trainingModelRows.size).pow(2))
        }
        return 1 - probabilities.sum()
    }

    private fun evaluatePredicate(p: Predicate<TrainingModeledRow<Classification>>): PredicateResult<Classification> {
        val resolvedAsTrue: MutableList<TrainingModeledRow<Classification>> = mutableListOf()
        val resolvedAsFalse: MutableList<TrainingModeledRow<Classification>> = mutableListOf()
        trainingModel.iterator().forEach { row: TrainingModeledRow<Classification> ->
            when (p.invoke(row)) {
                true -> resolvedAsTrue.add(row)
                false -> resolvedAsFalse.add(row)
            }
        }
        return PredicateResult(left = resolvedAsTrue.toList(), right = resolvedAsFalse.toList())
    }

    private fun calculateInformationGain(rows: List<TrainingModeledRow<Classification>>): List<InformationGain<Classification>> {
        val predicateInformationGain: MutableList<InformationGain<Classification>> = mutableListOf()
        @Suppress("UNCHECKED_CAST")
        val p = predicates as List<Predicate<TrainingModeledRow<Classification>>>
        p.iterator().forEach { predicate: Predicate<TrainingModeledRow<Classification>> ->
            val result: PredicateResult<Classification> = evaluatePredicate(predicate)
            val leftGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.left)
            val rightGiniImpurity: Double = this.calculateGiniImpurity(trainingModelRows = result.right)

            //weighted average of each question
            val avgImpurity: Double = (result.left.size.toDouble() / rows.size) * leftGiniImpurity
                    .plus(result.right.size.toDouble() / rows.size) * rightGiniImpurity
            predicateInformationGain.add(InformationGain(predicate, avgImpurity))
        }
        return predicateInformationGain.toList()

    }


    fun evaluate(row: TrainingModeledRow<Classification>): Classification {
        TODO("not implemented")
    }

    fun evaluate(rows: List<TrainingModeledRow<Classification>>): List<Classification> {
        TODO("not implemented")
    }
}


data class InformationGain<Classification>(val predicates: Predicate<TrainingModeledRow<Classification>>,
                                           val avgImpurity: Double)
