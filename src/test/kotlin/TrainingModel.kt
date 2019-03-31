import Diagnosis.*
import Symptom.*
import io.samuelagesilas.TrainingModeledRow

/*
DiagQuestion A	DiagQuestion B	Diagnosis
Sympton 1	    Sympton 2	    Diagnosis A
Sympton 1	    Sympton 3	    Diagnosis B
Sympton 4	    Sympton 5	    Diagnosis C
Sympton 1	    Sympton 3	    Diagnosis D
Sympton 1	    Sympton 5	    Diagnosis E
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
    const val Q1 = "Q1"
    const val Q2 = "Q2"
    const val Q3 = "Q3"
    const val Q4 = "Q4"
    const val Q5 = "Q5"
}

val trainingModel: List<TrainingModeledRow<Diagnosis>> = listOf(
    TestTrainingModelRow(Symptom1, Symptom2, DiagnosisA),
    TestTrainingModelRow(Symptom1, Symptom3, DiagnosisB),
    TestTrainingModelRow(Symptom4, Symptom5, DiagnosisC),
    TestTrainingModelRow(Symptom1, Symptom3, DiagnosisD),
    TestTrainingModelRow(Symptom1, Symptom5, DiagnosisE)
)