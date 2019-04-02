package io.samuelagesilas

import io.samuelagesilas.Diagnosis.*
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.pow

data class DataRow<T>(val diagnosisSymptom1: Symptom,
                      val diagnosisSymptom2: Symptom,
                      var diagnosis: Diagnosis? = null) : DecisionTreeClassifierDataRow<T>() {
    override fun classification(): T {
        @Suppress("UNCHECKED_CAST")
        return diagnosis as T
    }
}

class TrainingModelTest {

    @Test
    fun `test for correct rootGiniImpurity`() {
        val n: Double = 1 - listOf(
            (1.toDouble() / 8).pow(2),
            (2.toDouble() / 8).pow(2),
            (3.toDouble() / 8).pow(2),
            (1.toDouble() / 8).pow(2),
            (1.toDouble() / 8).pow(2)
        ).sum()
        Assertions.assertEquals(classifier.rootGiniImpurity, n)
    }

    @Test
    fun `test for correct order of predicateFunction labels`() {
        val i: Iterator<Predicate<Diagnosis>> = classifier.sortedPredicates.iterator()
        with(QuestionLabels) {
            assertEquals(this.Q4, i.next().predicateFunction.label)
            assertEquals(this.Q3, i.next().predicateFunction.label)
            assertEquals(this.Q1, i.next().predicateFunction.label)
            assertEquals(this.Q5, i.next().predicateFunction.label)
            assertEquals(this.Q2, i.next().predicateFunction.label)
        }
    }

    @Test
    fun `Test the avg impurity and information gain of Question 1`() {
        val impurity = object {
            val left: Double = 1 - listOf(
                (1.toDouble() / 6).pow(2),
                (2.toDouble() / 6).pow(2),
                (2.toDouble() / 6).pow(2),
                (1.toDouble() / 6).pow(2)
            ).sum()
            val right = 1 - listOf(
                (1.toDouble() / 2).pow(2),
                (1.toDouble() / 2).pow(2)
            ).sum()
        }
        val avgImpurity = (6.toDouble() / 8) * impurity.left + (2.toDouble() / 8) * impurity.right
        val p: Predicate<Diagnosis> = classifier.sortedPredicates.first { it.predicateFunction.label == "Question 1" }
        assertEquals(p.avgImpurity, avgImpurity)
        assertEquals(p.informationGain, classifier.rootGiniImpurity - avgImpurity)
    }

    @Test
    fun `Test the avg impurity and information gain of Question 4`() {
        val impurity = object {
            val left: Double = 1 - ((1.toDouble() / 1).pow(2))
            val right = 1 - listOf(
                (1.toDouble() / 7).pow(2),
                (1.toDouble() / 7).pow(2),
                (3.toDouble() / 7).pow(2),
                (1.toDouble() / 7).pow(2),
                (1.toDouble() / 7).pow(2)
            ).sum()
        }
        val avgImpurity = (1.toDouble() / 8) * impurity.left + (7.toDouble() / 8) * impurity.right
        val p: Predicate<Diagnosis> = classifier.sortedPredicates.first { it.predicateFunction.label == "Question 4" }
        assertEquals(p.avgImpurity, avgImpurity)
        assertEquals(p.informationGain, classifier.rootGiniImpurity - avgImpurity)
    }

    @Test
    fun `Test the avg impurity and information gain of Question 3`() {
        val impurity = object {
            val left: Double = 1 - listOf(
                (1.toDouble() / 3).pow(2),
                (1.toDouble() / 3).pow(2),
                (1.toDouble() / 3).pow(2)
            ).sum()
            val right = 1 - listOf(
                (1.toDouble() / 5).pow(2),
                (2.toDouble() / 5).pow(2),
                (2.toDouble() / 5).pow(2)
            ).sum()
        }
        val avgImpurity = (3.toDouble() / 8) * impurity.left + (5.toDouble() / 8) * impurity.right
        val p: Predicate<Diagnosis> = classifier.sortedPredicates.first { it.predicateFunction.label == "Question 3" }
        assertEquals(p.avgImpurity, avgImpurity)
        assertEquals(p.informationGain, classifier.rootGiniImpurity - avgImpurity)
    }

    @Test
    fun `test training data predicates with classifier`() {
        classifier.sortedPredicates.forEach { println("${it.predicateFunction.label}, ${it.avgImpurity}, ${it.informationGain}") }
        for (i in 0 until 100) {
            with(classifier.evaluate(trainingModel.first())) {
                assertTrue(this == DiagnosisA || this == DiagnosisB || this == DiagnosisC)
            }
        }
        assertEquals(classifier.evaluate(trainingModel[2]), DiagnosisC)
    }
}