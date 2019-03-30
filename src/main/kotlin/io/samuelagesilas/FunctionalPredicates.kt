package io.samuelagesilas

typealias Predicate<T> = (x: T) -> Boolean


data class PredicateResult<Classification>(val left: List<TrainingModeledRow<Classification>>,
                                           val right: List<TrainingModeledRow<Classification>>)