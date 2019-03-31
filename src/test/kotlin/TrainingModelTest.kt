import Diagnosis.*
import Symptom.*
import io.samuelagesilas.DecisionTreeClassifier
import io.samuelagesilas.PredicateFunction
import io.samuelagesilas.PredicateNode
import io.samuelagesilas.TrainingModeledRow
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

data class TestTrainingModelRow<T>(val diagnosisSymptom1: Symptom,
                                   val diagnosisSymptom2: Symptom,
                                   val diagnosis: Diagnosis) : TrainingModeledRow<T> {


    override fun classification(): T {
        @Suppress("UNCHECKED_CAST")
        return diagnosis as T
    }
}

class TrainingModelTest {

    @Test
    fun `test for correct training data`() {
        with(trainingModel[0] as TestTrainingModelRow) {
            assertEquals(Symptom1, this.diagnosisSymptom1)
            assertEquals(Symptom2, this.diagnosisSymptom2)
            assertEquals(DiagnosisA, this.classification())
        }
        with(trainingModel[1] as TestTrainingModelRow) {
            assertEquals(Symptom1, this.diagnosisSymptom1)
            assertEquals(Symptom3, this.diagnosisSymptom2)
            assertEquals(DiagnosisB, this.classification())
        }
        with(trainingModel[2] as TestTrainingModelRow) {
            assertEquals(Symptom4, this.diagnosisSymptom1)
            assertEquals(Symptom5, this.diagnosisSymptom2)
            assertEquals(DiagnosisC, this.classification())
        }
        with(trainingModel[3] as TestTrainingModelRow) {
            assertEquals(Symptom1, this.diagnosisSymptom1)
            assertEquals(Symptom3, this.diagnosisSymptom2)
            assertEquals(DiagnosisD, this.classification())
        }
        with(trainingModel[4] as TestTrainingModelRow) {
            assertEquals(Symptom1, diagnosisSymptom1)
            assertEquals(Symptom5, diagnosisSymptom2)
            assertEquals(DiagnosisE, this.classification())
        }

        val q1: PredicateFunction<TestTrainingModelRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q1) {
            it.diagnosisSymptom1 == Symptom.Symptom1 || it.diagnosisSymptom2 == Symptom.Symptom1
        }
        val q2: PredicateFunction<TestTrainingModelRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q2) {
            it.diagnosisSymptom1 == Symptom.Symptom2 || it.diagnosisSymptom2 == Symptom.Symptom2
        }
        val q3: PredicateFunction<TestTrainingModelRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q3) {
            it.diagnosisSymptom1 == Symptom.Symptom3 || it.diagnosisSymptom2 == Symptom.Symptom3
        }
        val q4: PredicateFunction<TestTrainingModelRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q4) {
            it.diagnosisSymptom1 == Symptom.Symptom4 || it.diagnosisSymptom2 == Symptom.Symptom4
        }
        val q5: PredicateFunction<TestTrainingModelRow<Diagnosis>> = PredicateFunction(label = QuestionLabels.Q5) {
            it.diagnosisSymptom1 == Symptom.Symptom5 || it.diagnosisSymptom2 == Symptom.Symptom5
        }

        //FunctionalPredicates<TestTrainingModelRow<Diagnosis>>
        val p: List<PredicateFunction<TestTrainingModelRow<Diagnosis>>> = listOf(q1, q2, q3, q4, q5)
        assertTrue(p[0].function.invoke(trainingModel[0] as TestTrainingModelRow<Diagnosis>))
        Assertions.assertTrue(p.get(0).function.invoke(trainingModel[1] as TestTrainingModelRow<Diagnosis>))
        assertFalse(p.get(1).function.invoke(trainingModel[4] as TestTrainingModelRow<Diagnosis>))


        val d: DecisionTreeClassifier<Diagnosis> = DecisionTreeClassifier(trainingModel = trainingModel,
                predicateFunctions = p)

        val i : Iterator<PredicateNode<Diagnosis>> = d.sortedPredicates.iterator()
        assertEquals("Q1", i.next().predicate.predicateFunction.label)
        assertEquals("Q2", i.next().predicate.predicateFunction.label)
        assertEquals("Q4", i.next().predicate.predicateFunction.label)
        assertEquals("Q3", i.next().predicate.predicateFunction.label)
        assertEquals("Q5", i.next().predicate.predicateFunction.label)

        println(d.rootGiniImpurity)
    }
}