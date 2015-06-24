/*
 *
 * ****************
 * Copyright 2015 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

import org.apache.spark.mllib.linalg.SparseVector;

import java.io.Serializable;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MultilabelPoint implements Serializable {

    private final int docID;

    /**
     * If available, The set of features representing this point.
     */
    private final SparseVector features;

    /**
     * The set of labels assigned to this point.
     */
    private final int[] labels;

    public MultilabelPoint(int docID, SparseVector features, int[] labels) {
        if (features == null)
            throw new NullPointerException("The set of features is 'null'");
        if (labels == null)
            throw new NullPointerException("The set of labels is 'null'");
        this.docID = docID;
        this.features = features;
        this.labels = labels;
    }

    /**
     * Get the set of features of this point.
     *
     * @return The set of features of this point.
     */
    public SparseVector getFeatures() {
        return features;
    }

    /**
     * Get the set of labels assigned to this point.
     *
     * @return The set of labels assigned to this point.
     */
    public int[] getLabels() {
        return labels;
    }

    public int getDocID() {
        return docID;
    }
}
