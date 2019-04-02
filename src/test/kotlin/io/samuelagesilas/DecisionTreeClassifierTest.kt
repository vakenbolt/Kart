package io.samuelagesilas

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import kotlin.math.pow

class DecisionTreeClassifierTest {
    @Test
    fun `test for correct rootGiniImpurity`() {
        val data: List<DataRow<Diagnosis>> = listOf(DataRow(Symptom.Symptom1, Symptom.Symptom2),
                                                    DataRow(Symptom.Symptom1, Symptom.Symptom3),
                                                    DataRow(Symptom.Symptom2, Symptom.Symptom5),
                                                    DataRow(Symptom.Symptom3, Symptom.Symptom1),
                                                    DataRow(Symptom.Symptom2, Symptom.Symptom5)
        )
        with(classifier.evaluate(listOf(data.first(), data[2]))) {
            assertEquals(this[0], Diagnosis.DiagnosisA)
            assertEquals(this[1], Diagnosis.DiagnosisC)
        }
        assertEquals(Diagnosis.DiagnosisC, classifier.evaluate(data[2]))
        with(classifier.evaluate(data[3])) {
            assertTrue(this == Diagnosis.DiagnosisB || this == Diagnosis.DiagnosisD)
        }
        assertEquals(Diagnosis.DiagnosisC, classifier.evaluate(data[4]))
    }
}