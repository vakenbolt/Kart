package io.samuelagesilas

import io.samuelagesilas.DiagnosisTestData.Diagnosis.*
import io.samuelagesilas.DiagnosisTestData.classifier
import io.samuelagesilas.DiagnosisTestData.trainingModel
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.pow

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
        Assertions.assertEquals(classifier.calculateGiniImpurity(trainingModel), n)
    }

    fun `test for correct order of predicateFunction labels`() {
        TODO("Implement this test by traversing the tree and comparing asserting against the expected order")
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
        val p = DecisionTreeClassifier(trainingModel, DiagnosisTestData.Questions.predicates)
            .calculateInformationGain(trainingModel)
            .first { it.predicateFunction.label == "Question 1" }
        assertEquals(p.avgImpurity, avgImpurity)
        assertEquals(p.informationGain, classifier.calculateGiniImpurity(trainingModel) - avgImpurity)
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
        val p = DecisionTreeClassifier(trainingModel, DiagnosisTestData.Questions.predicates)
            .calculateInformationGain(trainingModel)
            .first { it.predicateFunction.label == "Question 4" }
        assertEquals(p.avgImpurity, avgImpurity)
        assertEquals(p.informationGain, classifier.calculateGiniImpurity(trainingModel) - avgImpurity)
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
        val p = DecisionTreeClassifier(trainingModel, DiagnosisTestData.Questions.predicates)
            .calculateInformationGain(trainingModel)
            .first { it.predicateFunction.label == "Question 3" }
        assertEquals(p.avgImpurity, avgImpurity)
        assertEquals(p.informationGain, classifier.calculateGiniImpurity(trainingModel) - avgImpurity)
    }

    @Test
    fun `test training data predicates with classifier`() {
        for (i in 1..100) {
            with(classifier.evaluate(trainingModel.first())) {
                assertTrue(this == DiagnosisA || this == DiagnosisB || this == DiagnosisC)
            }
            with(classifier.evaluate(trainingModel[2])) {
                assertTrue(this == DiagnosisA || this == DiagnosisB || this == DiagnosisC)
            }
            assertEquals(DiagnosisD, classifier.evaluate(trainingModel[3]))
            assertEquals(DiagnosisE, classifier.evaluate(trainingModel[4]))
            assertEquals(DiagnosisB, classifier.evaluate(trainingModel[5]))
            with(classifier.evaluate(trainingModel[6])) {
                assertTrue(this == DiagnosisA || this == DiagnosisB || this == DiagnosisC)
            }
            assertEquals(DiagnosisC, classifier.evaluate(trainingModel[7]))
        }

    }
}