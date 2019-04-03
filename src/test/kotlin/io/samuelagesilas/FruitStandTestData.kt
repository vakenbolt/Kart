package io.samuelagesilas

import io.samuelagesilas.FruitStandTestData.Color.*
import io.samuelagesilas.FruitStandTestData.Fruit.*
import io.samuelagesilas.FruitStandTestData.QuestionLabels


data class FruitStandDataRow<T>(val color: FruitStandTestData.Color,
                                val diameter: Int,
                                var fruit: FruitStandTestData.Fruit? = null) : DecisionTreeClassifierDataRow<T>() {
    override fun classification(): T {
        @Suppress("UNCHECKED_CAST")
        return fruit as T
    }
}

object FruitStandTestData {
    enum class Color {
        Green,
        Yellow,
        Red,
    }

    enum class Fruit {
        Apple,
        Grape,
        Lemon,
    }

    object QuestionLabels {
        const val Q1 = "Question 1"
        const val Q2 = "Question 2"
        const val Q3 = "Question 3"
        const val Q4 = "Question 4"
        const val Q5 = "Question 5"
    }

    val trainingModel: List<DecisionTreeClassifierDataRow<Fruit>> = listOf(FruitStandDataRow(Green, 3, Apple),
                                                                           FruitStandDataRow(Yellow, 3, Apple),
                                                                           FruitStandDataRow(Red, 1, Grape),
                                                                           FruitStandDataRow(Red, 1, Grape),
                                                                           FruitStandDataRow(Yellow, 3, Lemon))

    object Questions {
        val q1: PredicateFunction<FruitStandDataRow<Fruit>> = PredicateFunction(label = QuestionLabels.Q1) {
            it.color == Green
        }
        val q2: PredicateFunction<FruitStandDataRow<Fruit>> = PredicateFunction(label = QuestionLabels.Q2) {
            it.diameter >= 3
        }
        val q3: PredicateFunction<FruitStandDataRow<Fruit>> = PredicateFunction(label = QuestionLabels.Q3) {
            it.color == Yellow
        }
        val q4: PredicateFunction<FruitStandDataRow<Fruit>> = PredicateFunction(label = QuestionLabels.Q4) {
            it.color == Red
        }
        val q5: PredicateFunction<FruitStandDataRow<Fruit>> = PredicateFunction(label = QuestionLabels.Q5) {
            it.diameter >= 1
        }

        val predicates: List<PredicateFunction<FruitStandDataRow<Fruit>>> = listOf(Questions.q1,
                                                                                   Questions.q2,
                                                                                   Questions.q3,
                                                                                   Questions.q4,
                                                                                   Questions.q5)
    }

    val classifier: DecisionTreeClassifier<Fruit> = DecisionTreeClassifier(FruitStandTestData.trainingModel,
                                                                           Questions.predicates)
}


