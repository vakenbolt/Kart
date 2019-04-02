package io.samuelagesilas

import io.samuelagesilas.Diagnosis.*
import io.samuelagesilas.Symptom.*

/*
DiagQuestion A   DiagQuestion B    Diagnosis
Symptom1         Symptom2          DiagnosisA
Symptom1         Symptom1          DiagnosisB
Symptom1         Symptom5          DiagnosisC
Symptom1         Symptom3          DiagnosisD
Symptom5         Symptom3          DiagnosisE
Symptom1         Symptom4          DiagnosisB
Symptom1         Symptom1          DiagnosisC
Symptom2         Symptom3          DiagnosisC
*/
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

val trainingModel: List<DecisionTreeClassifierDataRow<Diagnosis>> = listOf(DataRow(Symptom1, Symptom2, DiagnosisA),
                                                                           DataRow(Symptom1, Symptom1, DiagnosisB),
                                                                           DataRow(Symptom1, Symptom5, DiagnosisC),
                                                                           DataRow(Symptom1, Symptom3, DiagnosisD),
                                                                           DataRow(Symptom5, Symptom3, DiagnosisE),
                                                                           DataRow(Symptom1, Symptom4, DiagnosisB),
                                                                           DataRow(Symptom1, Symptom1, DiagnosisC),
                                                                           DataRow(Symptom2, Symptom3, DiagnosisC)
)

object Questions {
    val q1: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q1) {
        it.diagnosisSymptom1 == Symptom1 || it.diagnosisSymptom2 == Symptom5
    }
    val q2: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q2) {
        it.diagnosisSymptom1 == Symptom1 || it.diagnosisSymptom2 == Symptom3
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

    val predicates: List<PredicateFunction<DataRow<Diagnosis>>> = listOf(Questions.q1,
                                                                         Questions.q2,
                                                                         Questions.q3,
                                                                         Questions.q4,
                                                                         Questions.q5)
}


val classifier: DecisionTreeClassifier<Diagnosis> = DecisionTreeClassifier(trainingModel, Questions.predicates)