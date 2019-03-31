package io.samuelagesilas

/**
 * A predicate lambda identified as [function] and its associated [label].
 * [T] indicates the type associated with the classification column in the training model.
 */
data class PredicateFunction<T>(val label: String,
                                val function: (x: T) -> Boolean)


/**
 * The [left] and [right] side invocation result of a predicate lambda.
 * [T] indicates the type associated with the classification column in the training model.
 * @param left: A [List] of [TrainingModeledRow] if the invocation result of the predicate lambda is true
 * @param right: A [List] of [TrainingModeledRow] if the invocation result of the predicate lambda is false
 */
data class PredicateResult<T>(val left: List<TrainingModeledRow<T>>,
                              val right: List<TrainingModeledRow<T>>)


/**
 * The [predicateFunction] associated with the [avgImpurity] and [informationGain].
 * [T] indicates the type associated with the classification column in the training model.
 */
data class Predicate<T>(val predicateFunction: PredicateFunction<TrainingModeledRow<T>>,
                        val avgImpurity: Double,
                        val informationGain: Double)


/**
 * Node used in the [DecisionTreeClassifier] to build the decision tree.
 * [T] indicates the type associated with the classification column in the training model.
 */
data class PredicateNode<T>(var predicate: Predicate<T>,
                            var predicateIndex: Int?,
                            var parentNodeResults: List<TrainingModeledRow<T>>?,
                            var result: PredicateResult<T>? = null,
                            var leftNode: PredicateNode<T>? = null,
                            var rightNode: PredicateNode<T>? = null)