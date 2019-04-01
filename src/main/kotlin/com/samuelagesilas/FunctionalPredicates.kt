package com.samuelagesilas

/**
 * A predicate lambda identified as [function] and its associated [label].
 * [T] indicates the type associated with the classification column in the training model.
 */
data class PredicateFunction<T : DecisionTreeClassifierDataRow<*>>(val label: String,
                                                                                      val function: (row: T) -> Boolean)


/**
 * The [left] and [right] side invocation result of a predicate lambda.
 * [T] indicates the type associated with the classification column in the training model.
 * @param left: A [List] of [DecisionTreeClassifierDataRow] if the invocation result of the predicate lambda is true
 * @param right: A [List] of [DecisionTreeClassifierDataRow] if the invocation result of the predicate lambda is false
 */
data class PredicateResult<T>(val left: List<DecisionTreeClassifierDataRow<T>>,
                              val right: List<DecisionTreeClassifierDataRow<T>>)


/**
 * The [predicateFunction] associated with the [avgImpurity] and [informationGain].
 * [T] indicates the type associated with the classification column in the training model.
 */
data class Predicate<T>(val predicateFunction: PredicateFunction<DecisionTreeClassifierDataRow<T>>,
                        val avgImpurity: Double,
                        val informationGain: Double)


/**
 * Node used in the [DecisionTreeClassifier] to build the decision tree.
 * [T] indicates the type associated with the classification column in the training model.
 */
data class PredicateNode<T>(var predicate: Predicate<T>,
                            var predicateIndex: Int?,
                            var nodeResult: List<DecisionTreeClassifierDataRow<T>>?,
                            var result: PredicateResult<T>? = null,
                            var leftNode: PredicateNode<T>? = null,
                            var rightNode: PredicateNode<T>? = null)