package io.samuelagesilas

import io.samuelagesilas.DiagnosisTestData.Diagnosis
import io.samuelagesilas.DiagnosisTestData.Symptom
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class DecisionTreeClassifierTest {
    @Test
    fun `test evaluate method`() {
        val data: List<DiagnosisDataRow<Diagnosis>> = listOf(DiagnosisDataRow(Symptom.Symptom1, Symptom.Symptom2),
                                                             DiagnosisDataRow(Symptom.Symptom1, Symptom.Symptom3),
                                                             DiagnosisDataRow(Symptom.Symptom2, Symptom.Symptom5),
                                                             DiagnosisDataRow(Symptom.Symptom3, Symptom.Symptom1),
                                                             DiagnosisDataRow(Symptom.Symptom2, Symptom.Symptom4))
        for (i in 1..100) {
            with(DiagnosisTestData.classifier.evaluate(listOf(data.first(), data[2]))) {
                this.forEach {
                    assertTrue(it == Diagnosis.DiagnosisA
                                       || it == Diagnosis.DiagnosisB
                                       || it == Diagnosis.DiagnosisC)
                }

            }
            with (DiagnosisTestData.classifier) {
                assertEquals(Diagnosis.DiagnosisD, this.evaluate(data[1]))
                assertEquals(Diagnosis.DiagnosisC, this.evaluate(data[3]))
                assertEquals(Diagnosis.DiagnosisB, this.evaluate(data[4]))
            }
        }
    }
}