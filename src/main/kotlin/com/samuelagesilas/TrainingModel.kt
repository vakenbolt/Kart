package com.samuelagesilas

/**
 * Interface for any data type used as a row in a training model.
 * [T] indicates the type associated with the classification column in the training model.
 */
abstract class DecisionTreeClassifierDataRow<T> {

    /**
     * Returns the typed value associated with the classification column in the training model.
     */
    open fun classification(): T? {
        return null
    }
}
