/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ******************
 */

package it.tizianofagni.sparkboost;

import java.io.Serializable;

/**
 * The set of results obtained from a classification task performed by
 * {@link BoostClassifier} class.
 * <p>
 * Following is an example of iteration over classification results:<br/>
 * <pre>
 * StringBuilder sb = new StringBuilder();
 * sb.append("**** Effectiveness\n");
 * sb.append(results.getCt().toString() + "\n");
 * sb.append("********\n");
 * for (int i = 0; i < results.getNumDocs(); i++) {
 * int docID = results.getDocuments()[i];
 * int[] labels = results.getLabels()[i];
 * int[] goldLabels = results.getGoldLabels()[i];
 * sb.append("DocID: " + docID + ", Labels assigned: " +
 *      Arrays.toString(labels) + ", Labels scores: " +
 *      Arrays.toString(results.getScores()[i]) +
 *      ", Gold labels: " + Arrays.toString(goldLabels) + "\n");
 * }
 * </pre>
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class ClassificationResults implements Serializable {
    private final int numDocs;
    private final int[] documents;
    private final int[][] labels;
    private final double[][] scores;
    private final int[][] goldLabels;
    private final ContingencyTable ct;

    public ClassificationResults(int numDocs, int[] documents, int[][] labels, double[][] scores, int[][] goldLabels, ContingencyTable ct) {
        this.numDocs = numDocs;
        this.documents = documents;
        this.labels = labels;
        this.scores = scores;
        this.goldLabels = goldLabels;
        this.ct = ct;
    }

    /**
     * Get the number of documents classified in this results set.
     *
     * @return The number of documents classified in this results set.
     */
    public int getNumDocs() {
        return numDocs;
    }

    /**
     * Get the scores obtained for each label automatically assigned to a specific document. The greater is
     * the value of the score, the greater is the strength of the decision of the classifier in assigning that
     * specific label to a document.
     *
     * @return The scores obtained for each label automatically assigned to a specific document.
     */
    public double[][] getScores() {
        return scores;
    }

    /**
     * Get the label IDs assigned to the documents.
     *
     * @return The label IDs assigned to the documents.
     */
    public int[][] getLabels() {
        return labels;
    }

    /**
     * Get the contingency table computed for this classification task.
     *
     * @return The contingency table computed for this classification task.
     */
    public ContingencyTable getCt() {
        return ct;
    }

    /**
     * Get the gold labels IDs (i.e. the set of labels originally assigned to a document) for each classified document. If
     * the gold labels are not available, the set of labels specific for a specific document will be empty.
     *
     * @return The gold labels IDs (i.e. the set of labels originally assigned to a document) for each classified document
     */
    public int[][] getGoldLabels() {
        return goldLabels;
    }

    /**
     * Return the document IDs of documents classified.
     *
     * @return The document IDs of documents classified.
     */
    public int[] getDocuments() {
        return documents;
    }
}
