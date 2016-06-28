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

import org.apache.spark.mllib.linalg.SparseVector;

import java.io.Serializable;

/**
 * This is the representation of a point or document in a multiclass/binary
 * problem.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MultilabelPoint implements Serializable {

    /**
     * The document unique ID.
     */
    private final int pointID;

    /**
     * The set of features representing this point.
     */
    private final SparseVector features;

    /**
     * The set of labels assigned to this point or an empty set if no labels are assigned.
     */
    private final int[] labels;

    public MultilabelPoint(int pointID, SparseVector features, int[] labels) {
        if (features == null)
            throw new NullPointerException("The set of features is 'null'");
        if (labels == null)
            throw new NullPointerException("The set of labels is 'null'");
        this.pointID = pointID;
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
     * Get the set of labels assigned to this point. In binary problems, a point can have assigned
     * at most 1 label (labelID equals to 0).
     *
     * @return The set of labels assigned to this point or an empty set if no labels are assigned to it.
     */
    public int[] getLabels() {
        return labels;
    }

    /**
     * Get the point unique ID. Every point in a {@link org.apache.spark.api.java.JavaRDD} must have
     * an unique assigned ID.
     *
     * @return The point unique ID.
     */
    public int getPointID() {
        return pointID;
    }
}
