package io.samuelagesilas

data class PredicateFunction<T>(val label: String,
                                val function: (x: T) -> Boolean)


data class PredicateResult<Classification>(val left: List<TrainingModeledRow<Classification>>,
                                           val right: List<TrainingModeledRow<Classification>>)


/**
 * Data class indicating the [predicateFunction] associated with the [avgImpurity] and [informationGain].
 * [T] indicates the type associated with the classification column in the training model.
 */
data class Predicate<T>(val predicateFunction: PredicateFunction<TrainingModeledRow<T>>,
                        val avgImpurity: Double,
                        val informationGain: Double)


data class PredicateNode<T>(val predicate: Predicate<T>,
                            var leftNode: PredicateNode<T>? = null,
                            var rightNode: PredicateNode<T>? = null) {

}