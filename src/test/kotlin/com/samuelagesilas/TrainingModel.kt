package com.samuelagesilas

import com.samuelagesilas.Diagnosis.*
import com.samuelagesilas.Symptom.*

/*
DiagQuestion A   DiagQuestion B    com.samuelagesilas.Diagnosis
Sympton 1        Sympton 2         com.samuelagesilas.Diagnosis A
Sympton 1        Sympton 3         com.samuelagesilas.Diagnosis B
Sympton 4        Sympton 5         com.samuelagesilas.Diagnosis C
Sympton 1        Sympton 3         com.samuelagesilas.Diagnosis D
Sympton 1        Sympton 5         com.samuelagesilas.Diagnosis E
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

val trainingModel: List<DecisionTreeClassifierRow<Diagnosis>> = listOf(DataRow(Symptom1, Symptom2, DiagnosisA),
                                                                       DataRow(Symptom1, Symptom3, DiagnosisB),
                                                                       DataRow(Symptom4, Symptom5, DiagnosisC),
                                                                       DataRow(Symptom1, Symptom3, DiagnosisD),
                                                                       DataRow(Symptom1, Symptom5, DiagnosisE)
)