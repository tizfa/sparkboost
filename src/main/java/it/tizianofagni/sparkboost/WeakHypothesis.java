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

import java.io.Serializable;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class WeakHypothesis implements Serializable {
    private final WeakHypothesisData[] labelsHypothesis;


    public WeakHypothesis(int numLabels) {
        labelsHypothesis = new WeakHypothesisData[numLabels];
    }

    public void setLabelData(int labelID, WeakHypothesisData whd) {
        if (labelID < 0 || labelID >= labelsHypothesis.length)
            throw new IllegalArgumentException("The label ID is not valid: " + labelID);
        if (whd == null)
            throw new NullPointerException("The weak hypothesis data is 'nuyll'");
        if (labelID != whd.getLabelID())
            throw new IllegalArgumentException("The label ID specified in weak hypothesis data is different!");
        labelsHypothesis[labelID] = whd;
    }

    public WeakHypothesisData getLabelData(int labelID) {
        if (labelID < 0 || labelID >= labelsHypothesis.length)
            throw new IllegalArgumentException("The label ID is not valid: " + labelID);
        return labelsHypothesis[labelID];
    }

    public int getNumLabels() {
        return labelsHypothesis.length;
    }

    public static class WeakHypothesisData implements Serializable {

        /**
         * The label ID.
         */
        private final int labelID;

        /**
         * The ID of the feature "pivot" for this label in this hypothesis.
         */
        private final int featureID;

        /**
         * The computed C0 value.
         */
        private final double c0;

        /**
         * The computed C1 value.
         */
        private final double c1;

        public WeakHypothesisData(int labelID, int featureID, double c0, double c1) {
            this.labelID = labelID;
            this.featureID = featureID;
            this.c0 = c0;
            this.c1 = c1;
        }

        public int getLabelID() {
            return labelID;
        }

        public int getFeatureID() {
            return featureID;
        }

        public double getC0() {
            return c0;
        }

        public double getC1() {
            return c1;
        }
    }
}
