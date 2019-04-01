<img src="https://svgsilh.com/png-512/1296971.png" alt="alt text" width="100" height="100">

## Kart: Decision Tree Classifier for Kotlin


Cart-based Machine Learning algorithm implemented using the standard Java and Kotlin libraries.

##### Getting Started:

Start by creating a training data set that is used by the classifier to analyze and build the decision tree.

Diagnosis Question 1 | Diagnosis Question 2 | Diagnosis
------------ | ------------- | -------------
Sympton 1 | Sympton 2 | Diagnosis A
Sympton 1 | Sympton 3 | Diagnosis B
Sympton 4 | Sympton 5 | Diagnosis C
Sympton 1 | Sympton 3 | Diagnosis D
Sympton 1 | Sympton 5 | Diagnosis E

Use a data class(if needed, a regular class will also work) to create a typed representation of a row in the training data set. The resulting class must implement the `DecisionTreeClassifierDataRow` interface. The `classification` method returns the value of the classification column.
>In the example below, the classification column is the `diagnosis` field whose type is the `Diagnosis` enum.
```kotlin
data class DataRow<T>(
    val diagnosisSymptom1: Symptom,
    val diagnosisSymptom2: Symptom,
    var diagnosis: Diagnosis? = null) : DecisionTreeClassifierDataRow<T>() {
    override fun classification(): T {
        @Suppress("UNCHECKED_CAST")
        return diagnosis as T
    }
}
```

The training data is created as a `List` of objects that implement the `DecisionTreeClassifierDataRow<T>` interface where `<T>` indicates the type associated with the classification column in the training model.
```kotlin
List<DecisionTreeClassifierDataRow<Diagnosis>> = listOf(DataRow(Symptom1, Symptom2, DiagnosisA),
                                                        DataRow(Symptom1, Symptom3, DiagnosisB),
                                                        DataRow(Symptom4, Symptom5, DiagnosisC),
                                                        DataRow(Symptom1, Symptom3, DiagnosisD),
                                                        DataRow(Symptom1, Symptom5, DiagnosisE)
```

The predicate's or _questions_ used to analyze the training data is done with the `PredicateFunction` class which takes a label and lambda as the predicate function.
```kotlin
val q1: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = "q1") {
    it.diagnosisSymptom1 == Symptom1 || it.diagnosisSymptom2 == Symptom1
}
val q2: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = "q2") {
    it.diagnosisSymptom1 == Symptom2 || it.diagnosisSymptom2 == Symptom2
}
val q3: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = "q3") {
    it.diagnosisSymptom1 == Symptom3 || it.diagnosisSymptom2 == Symptom3
}
val q4: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = "q4") {
    it.diagnosisSymptom1 == Symptom4 || it.diagnosisSymptom2 == Symptom4
}
val q5: PredicateFunction<DataRow<Diagnosis>> = PredicateFunction(label = "q5") {
    it.diagnosisSymptom1 == Symptom5 || it.diagnosisSymptom2 == Symptom5
}
```

Now we can create our decision tree, providing a list of predicates and the associated data model. Once the `DecisionTreeClassifier` class is instantiated it automatically makes the necessary calculations and builds the appropriate decision tree for data processing.
```kotlin
val p: List<PredicateFunction<DataRow<Diagnosis>>> = listOf(q1, q2, q3, q4, q5)
val classifier: DecisionTreeClassifier<Diagnosis> = DecisionTreeClassifier(
     trainingModel = trainingModel,
     predicateFunctions = p)
```

Here is a sample list of data provided to the classifier for analysis.
```kotlin
val data: List<DataRow<Diagnosis>> = listOf(
    DataRow(Symptom1, Symptom2),
    DataRow(Symptom1, Symptom3),
    DataRow(Symptom2, Symptom5),
    DataRow(Symptom3, Symptom1),
    DataRow(Symptom2, Symptom5)
)
```

To evaluate and retrieve the classification for a _row_ of data.
```kotlin
classifier.evaluate(data.first())

Returns:
DiagnosisA
```


In this example, the response from the classifier will either `DiagnosisB` or `DiagnosisD` because the provided questions associated with the given training data could not be partitioned further.
```kotlin
classifier.evaluate(data[3])

Returns:
DiagnosisB or DiagnosisD
```