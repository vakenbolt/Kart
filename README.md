<img src="https://svgsilh.com/png-512/1296971.png" alt="alt text" height="150">

## Kart: Decision Tree Classifier for Kotlin


Cart-based Machine Learning algorithm implemented using the standard Java and Kotlin libraries.

##### Getting Started:

Start by creating a training data set that is used by the classifier to analyze and build the decision tree.

Diagnosis Question 1 | Diagnosis Question 2 | Diagnosis
------------ | ------------- | -------------
Symptom1 | Symptom2 | DiagnosisA
Symptom1 | Symptom1 | DiagnosisB
Symptom1 | Symptom5 | DiagnosisC
Symptom1 | Symptom3 | DiagnosisD
Symptom5 | Symptom3 | DiagnosisE
Symptom1 | Symptom4 | DiagnosisB
Symptom1 | Symptom1 | DiagnosisC
Symptom2 | Symptom3 | DiagnosisC

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
                                                        DataRow(Symptom1, Symptom1, DiagnosisB),
                                                        DataRow(Symptom1, Symptom5, DiagnosisC),
                                                        DataRow(Symptom1, Symptom3, DiagnosisD),
                                                        DataRow(Symptom5, Symptom3, DiagnosisE),
                                                        DataRow(Symptom1, Symptom4, DiagnosisB),
                                                        DataRow(Symptom1, Symptom1, DiagnosisC),
                                                        DataRow(Symptom2, Symptom3, DiagnosisC))
                                                        
                                                        
```

The predicate's or _questions_ used to analyze the training data is done with the `PredicateFunction` class which takes a label and lambda as the predicate function.
```kotlin
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
    DataRow(Symptom1, Symptom1))
```

To evaluate and retrieve the classification for a _row_ of data.
```kotlin
classifier.evaluate(data[2])
```

Returns:
```
DiagnosisC
```


In this example, the response from the classifier will either `DiagnosisA`, `DiagnosisB` or `DiagnosisC` because the provided questions associated with the given training data could not be partitioned further.
```kotlin
classifier.evaluate(data.first())
```

Returns:
```
DiagnosisA, DiagnosisB or DiagnosisC
```