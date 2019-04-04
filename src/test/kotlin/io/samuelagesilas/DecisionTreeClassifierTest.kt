package io.samuelagesilas

import io.samuelagesilas.DiagnosisTestData.Diagnosis
import io.samuelagesilas.DiagnosisTestData.Symptom
import io.samuelagesilas.FruitStandTestData.Fruit
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class DecisionTreeClassifierTest {
    companion object {
        fun <T> assertNodeIsNull(node: PredicateNode<T>) {
            Assertions.assertNull(node.leftNode)
            Assertions.assertNull(node.rightNode)
            Assertions.assertNull(node.result)
            Assertions.assertNull(node.predicateFunction)
        }
    }

    @Test
    fun `test diagnosis with custom data`() {
        val data: List<DiagnosisDataRow<Diagnosis>> = listOf(DiagnosisDataRow(Symptom.Symptom1, Symptom.Symptom2),
                                                             DiagnosisDataRow(Symptom.Symptom1, Symptom.Symptom3),
                                                             DiagnosisDataRow(Symptom.Symptom2, Symptom.Symptom5),
                                                             DiagnosisDataRow(Symptom.Symptom3, Symptom.Symptom1),
                                                             DiagnosisDataRow(Symptom.Symptom2, Symptom.Symptom4),
                                                             DiagnosisDataRow(Symptom.Symptom5, Symptom.Symptom3))
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
                assertEquals(Diagnosis.DiagnosisE, this.evaluate(data[5]))
            }
        }
    }

    @Test
    fun `test fruit stand`() {
        val m : List<DecisionTreeClassifierDataRow<Fruit>> = FruitStandTestData.trainingModel
        for (i in 1..100) {
            assertEquals(Fruit.Apple, FruitStandTestData.classifier.evaluate(m[0]))
            with (FruitStandTestData.classifier.evaluate(m[1])) {
                assertTrue(this == Fruit.Apple || this == Fruit.Lemon)
            }
            assertEquals(Fruit.Grape, FruitStandTestData.classifier.evaluate(m[2]))
            assertEquals(Fruit.Grape, FruitStandTestData.classifier.evaluate(m[3]))
            with (FruitStandTestData.classifier.evaluate(m[4])) {
                assertTrue(this == Fruit.Apple || this == Fruit.Lemon)
            }
        }
    }

    @Test
    fun `test diagnosis decision tree structure`() {
        with(DecisionTreeNodeBuilder(DiagnosisTestData.classifier)) {
            val decisionTree: PredicateNode<Diagnosis> = this.buildDecisionTree()
            assertEquals("Question 4", decisionTree.predicateFunction!!.label)
            assertEquals(7, decisionTree.leftNode!!.nodeResult!!.size)
            assertEquals(1, decisionTree.rightNode!!.nodeResult!!.size)

            assertNodeIsNull(decisionTree.rightNode!!)

            val node1L: PredicateNode<Diagnosis> = decisionTree.leftNode!!
            assertEquals("Question 3", node1L.predicateFunction!!.label)
            assertNotNull(node1L.leftNode)
            assertNotNull(node1L.rightNode)
            assertEquals(7, node1L.nodeResult!!.size)
            assertEquals(4, node1L.leftNode!!.nodeResult!!.size)
            assertEquals(3, node1L.rightNode!!.nodeResult!!.size)

            val node2R: PredicateNode<Diagnosis> = node1L.rightNode!!
            assertEquals("Question 1", node2R.predicateFunction!!.label)
            assertEquals(3, node2R.nodeResult!!.size)
            assertEquals(2, node2R.leftNode!!.nodeResult!!.size)
            assertEquals(1, node2R.rightNode!!.nodeResult!!.size)

            assertNodeIsNull(node2R.rightNode!!)

            val node3L: PredicateNode<Diagnosis> = node2R.leftNode!!
            assertEquals("Question 5", node3L.predicateFunction!!.label)
            assertEquals(1, node3L.leftNode!!.nodeResult!!.size)
            assertEquals(1, node3L.rightNode!!.nodeResult!!.size)

            assertNodeIsNull(node3L.leftNode!!)
            assertNodeIsNull(node3L.rightNode!!)
        }
    }
}