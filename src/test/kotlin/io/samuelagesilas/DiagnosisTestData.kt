package io.samuelagesilas

import io.samuelagesilas.DiagnosisTestData.Diagnosis.*
import io.samuelagesilas.DiagnosisTestData.Symptom.*


data class DiagnosisDataRow<T>(val diagnosisSymptom1: DiagnosisTestData.Symptom,
                               val diagnosisSymptom2: DiagnosisTestData.Symptom,
                               var diagnosis: DiagnosisTestData.Diagnosis? = null) :
    DecisionTreeClassifierDataRow<T>() {
    override fun classification(): T {
        @Suppress("UNCHECKED_CAST")
        return diagnosis as T
    }
}

object DiagnosisTestData {
    enum class Symptom {
        Symptom1,
        Symptom2,
        Symptom3,
        Symptom4,
        Symptom5,
    }

    enum class Diagnosis {
        DiagnosisA,
        DiagnosisB,
        DiagnosisC,
        DiagnosisD,
        DiagnosisE,
    }

    object QuestionLabels {
        const val Q1 = "Question 1"
        const val Q2 = "Question 2"
        const val Q3 = "Question 3"
        const val Q4 = "Question 4"
        const val Q5 = "Question 5"
    }

    val trainingModel: List<DecisionTreeClassifierDataRow<Diagnosis>> = listOf(
        DiagnosisDataRow(Symptom1, Symptom2, DiagnosisA),
        DiagnosisDataRow(Symptom1, Symptom1, DiagnosisB),
        DiagnosisDataRow(Symptom1, Symptom5, DiagnosisC),
        DiagnosisDataRow(Symptom1, Symptom3, DiagnosisD),
        DiagnosisDataRow(Symptom5, Symptom3, DiagnosisE),
        DiagnosisDataRow(Symptom1, Symptom4, DiagnosisB),
        DiagnosisDataRow(Symptom1, Symptom1, DiagnosisC),
        DiagnosisDataRow(Symptom2, Symptom3, DiagnosisC))

    object Questions {
        val q1: PredicateFunction<DiagnosisDataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q1) {
            it.diagnosisSymptom1 == Symptom1 || it.diagnosisSymptom2 == Symptom5
        }
        val q2: PredicateFunction<DiagnosisDataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q2) {
            it.diagnosisSymptom1 == Symptom1 || it.diagnosisSymptom2 == Symptom3
        }
        val q3: PredicateFunction<DiagnosisDataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q3) {
            it.diagnosisSymptom1 == Symptom3 || it.diagnosisSymptom2 == Symptom3
        }
        val q4: PredicateFunction<DiagnosisDataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q4) {
            it.diagnosisSymptom1 == Symptom4 || it.diagnosisSymptom2 == Symptom4
        }
        val q5: PredicateFunction<DiagnosisDataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q5) {
            it.diagnosisSymptom1 == Symptom5 || it.diagnosisSymptom2 == Symptom5
        }

        val predicates: List<PredicateFunction<DiagnosisDataRow<Diagnosis>>> = listOf(Questions.q1,
                                                                                      Questions.q2,
                                                                                      Questions.q3,
                                                                                      Questions.q4,
                                                                                      Questions.q5)
    }

    val classifier: DecisionTreeClassifier<Diagnosis> = DecisionTreeClassifier(trainingModel, Questions.predicates)
}