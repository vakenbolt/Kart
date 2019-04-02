package io.samuelagesilas

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class DecisionTreeClassifierTest {
    @Test
    fun `test evaluate method`() {
        val data: List<DataRow<Diagnosis>> = listOf(DataRow(Symptom.Symptom1, Symptom.Symptom2),
                                                    DataRow(Symptom.Symptom1, Symptom.Symptom3),
                                                    DataRow(Symptom.Symptom2, Symptom.Symptom5),
                                                    DataRow(Symptom.Symptom3, Symptom.Symptom1),
                                                    DataRow(Symptom.Symptom2, Symptom.Symptom4))
        for (i in 1..100) {
            with(classifier.evaluate(listOf(data.first(), data[2]))) {
                assertTrue(this[0] == Diagnosis.DiagnosisA
                                   || this[0] == Diagnosis.DiagnosisB
                                   || this[0] == Diagnosis.DiagnosisC)
                assertEquals(this[1], Diagnosis.DiagnosisC)
            }
            assertEquals(Diagnosis.DiagnosisC, classifier.evaluate(data[2]))
            assertEquals(Diagnosis.DiagnosisC, classifier.evaluate(data[3]))
            assertEquals(Diagnosis.DiagnosisB, classifier.evaluate(data[4]))
        }
    }
}