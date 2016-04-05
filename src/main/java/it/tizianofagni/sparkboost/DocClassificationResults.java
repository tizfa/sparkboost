

/*
 *
 * ****************
 * This file is part of sparkboost software package (https://github.com/tizfa/sparkboost).
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
 * {@link BoostClassifier} class over a single document available in the test set.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class DocClassificationResults implements Serializable {
    private final int docID;
    private final int[] labels;
    private final double[] scores;
    private final int[] goldLabels;
    private final ContingencyTable ct;

    public DocClassificationResults(int docID, int[] labels, double[] scores, int[] goldLabels, ContingencyTable ct) {
        this.docID = docID;
        this.labels = labels;
        this.scores = scores;
        this.goldLabels = goldLabels;
        this.ct = ct;
    }


    /**
     * Get the scores obtained for each label automatically assigned to a specific document. The greater is
     * the value of the score, the greater is the strength of the decision of the classifier in assigning that
     * specific label to a document.
     *
     * @return The scores obtained for each label automatically assigned to a specific document.
     */
    public double[] getScores() {
        return scores;
    }

    /**
     * Get the label IDs assigned to the documents.
     *
     * @return The label IDs assigned to the documents.
     */
    public int[] getLabels() {
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
    public int[] getGoldLabels() {
        return goldLabels;
    }

    /**
     * Return the document ID.
     *
     * @return The document ID.
     */
    public int getDocID() {
        return docID;
    }
}
