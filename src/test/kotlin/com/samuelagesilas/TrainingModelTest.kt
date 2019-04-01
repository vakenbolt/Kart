package com.samuelagesilas

import com.samuelagesilas.Diagnosis.*
import com.samuelagesilas.Symptom.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.math.pow

data class DataRow<T>(val diagnosisSymptom1: Symptom,
                      val diagnosisSymptom2: Symptom,
                      var diagnosis: Diagnosis? = null) : DecisionTreeClassifierRow<T>() {
    override fun classification(): T {
        @Suppress("UNCHECKED_CAST")
        return diagnosis as T
    }
}

class TrainingModelTest {

    @Test
    fun `test for correct training data`() {
        with(trainingModel[0] as DataRow) {
            assertEquals(Symptom1, this.diagnosisSymptom1)
            assertEquals(Symptom2, this.diagnosisSymptom2)
            assertEquals(DiagnosisA, this.classification())
        }
        with(trainingModel[1] as DataRow) {
            assertEquals(Symptom1, this.diagnosisSymptom1)
            assertEquals(Symptom3, this.diagnosisSymptom2)
            assertEquals(DiagnosisB, this.classification())
        }
        with(trainingModel[2] as DataRow) {
            assertEquals(Symptom4, this.diagnosisSymptom1)
            assertEquals(Symptom5, this.diagnosisSymptom2)
            assertEquals(DiagnosisC, this.classification())
        }
        with(trainingModel[3] as DataRow) {
            assertEquals(Symptom1, this.diagnosisSymptom1)
            assertEquals(Symptom3, this.diagnosisSymptom2)
            assertEquals(DiagnosisD, this.classification())
        }
        with(trainingModel[4] as DataRow) {
            assertEquals(Symptom1, diagnosisSymptom1)
            assertEquals(Symptom5, diagnosisSymptom2)
            assertEquals(DiagnosisE, this.classification())
        }

        val q1: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q1) {
            it.diagnosisSymptom1 == Symptom1 || it.diagnosisSymptom2 == Symptom1
        }
        val q2: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q2) {
            it.diagnosisSymptom1 == Symptom2 || it.diagnosisSymptom2 == Symptom2
        }
        val q3: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q3) {
            it.diagnosisSymptom1 == Symptom3 || it.diagnosisSymptom2 == Symptom3
        }
        val q4: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q4) {
            it.diagnosisSymptom1 == Symptom4 || it.diagnosisSymptom2 == Symptom4
        }
        val q5: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q5) {
            it.diagnosisSymptom1 == Symptom5 || it.diagnosisSymptom2 == Symptom5
        }

        //FunctionalPredicates<com.samuelagesilas.DataRow<com.samuelagesilas.Diagnosis>>
        val p: List<PredicateFunction<DataRow<Diagnosis>>> = listOf(q1, q2, q3, q4, q5)
//        assertTrue(p[0].function.invoke(com.samuelagesilas.getTrainingModel[0] as com.samuelagesilas.DataRow<com.samuelagesilas.Diagnosis>))
//        Assertions.assertTrue(p.get(0).function.invoke(com.samuelagesilas.getTrainingModel[1] as com.samuelagesilas.DataRow<com.samuelagesilas.Diagnosis>))
//        assertFalse(p.get(1).function.invoke(com.samuelagesilas.getTrainingModel[4] as com.samuelagesilas.DataRow<com.samuelagesilas.Diagnosis>))
//

        val d: DecisionTreeClassifier<Diagnosis> = DecisionTreeClassifier(trainingModel = trainingModel,
                                                                          predicateFunctions = p)
        d.sortedPredicates.forEach { println("${it.predicateFunction.label}, ${it.avgImpurity}, ${it.informationGain}") }

        val i: Iterator<Predicate<Diagnosis>> = d.sortedPredicates.iterator()
        assertEquals(QuestionLabels.Q1, i.next().predicateFunction.label)
        assertEquals(QuestionLabels.Q2, i.next().predicateFunction.label)
        assertEquals(QuestionLabels.Q4, i.next().predicateFunction.label)
        assertEquals(QuestionLabels.Q3, i.next().predicateFunction.label)
        assertEquals(QuestionLabels.Q5, i.next().predicateFunction.label)

        println(d.rootGiniImpurity)

        println(d.evaluate(trainingModel.first()))
        println(d.evaluate(trainingModel[2]))

        println("----")
        val t: List<DataRow<Diagnosis>> = listOf(DataRow(Symptom1, Symptom2),
                                                 DataRow(Symptom1, Symptom3),
                                                 DataRow(Symptom4, Symptom5),
                                                 DataRow(Symptom1, Symptom3),
                                                 DataRow(Symptom1, Symptom5)
        )
        println(d.evaluate(listOf(t.first(), t[2])))
        val n = 1 - listOf((1.toDouble() / 5).pow(2),
                           (1.toDouble() / 5).pow(2),
                           (1.toDouble() / 5).pow(2),
                           (1.toDouble() / 5).pow(2),
                           (1.toDouble() / 5).pow(2)).sum()
        assertEquals(d.rootGiniImpurity, n)
    }
}