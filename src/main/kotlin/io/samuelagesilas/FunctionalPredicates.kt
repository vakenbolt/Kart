package io.samuelagesilas

typealias Predicate<T> = (x: T) -> Boolean

/*
interface Predicates<T : TrainingModeledRow<*>> {
    fun addPredicate(f: Predicate<T>): Int
    fun getPredicate(i: Int): Predicate<T>
    fun size(): Int
    fun iterator(): Iterator<Predicate<T>>
}

fun <T>foo(foo: List<TrainingModeledRow<T>>) {

}

class FunctionalPredicates<T : TrainingModeledRow<*>> : Predicates<T> {

    private val foo: List<TrainingModeledRow<T>> ? = null
    private val _predicates: MutableList<Predicate<T>> = mutableListOf()

    override fun addPredicate(f: Predicate<T>): Int = with(this._predicates) {
        this.add(f)
        return this.size - 1
    }

    override fun getPredicate(i: Int): Predicate<T> = this._predicates[i]

    override fun size(): Int = _predicates.size

    override fun iterator(): Iterator<Predicate<T>> = this._predicates.toList().iterator()
}
*/

data class PredicateResult<Classification>(val left: List<TrainingModeledRow<Classification>>,
                                           val right: List<TrainingModeledRow<Classification>>)