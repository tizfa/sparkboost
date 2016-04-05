

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
 * A contingency table for a classification task.<br/><br/>
 * This structure allows to evaluate a classification
 * task by providing some specific measures like {@link #f1()},
 * {@link #precision()} or {@link #recall()}.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class ContingencyTable implements Serializable {

    private final int tp;
    private final int tn;
    private final int fp;
    private final int fn;


    public ContingencyTable(int tp, int tn, int fp, int fn) {
        this.tp = tp;
        this.tn = tn;
        this.fp = fp;
        this.fn = fn;
    }


    /**
     * Get the number of true positives.
     *
     * @return The number of true positives.
     */
    public int tp() {
        return tp;
    }

    /**
     * Get the number of true negatives.
     *
     * @return The number of true negatives.
     */
    public int tn() {
        return tn;
    }

    /**
     * Get the number of false positives.
     *
     * @return The number of false positives.
     */
    public int fp() {
        return fp;
    }

    /**
     * Get the number of false negatives.
     *
     * @return The number of false negatives.
     */
    public int fn() {
        return fn;
    }


    /**
     * Get the obtained precision.
     *
     * @return The obtained precision.
     */
    public double precision() {
        double den = tp + fp;
        if (den != 0)
            return tp / den;
        else
            return 1.0;
    }

    /**
     * Get the obtained recall.
     *
     * @return The obtained recall.
     */
    public double recall() {
        double den = tp + fn;
        if (den != 0)
            return tp / den;
        else
            return 1.0;
    }

    /**
     * Get the F_beta measure.
     *
     * @param beta The beta param value.
     * @return The F_beta measure.
     */
    public double f(double beta) {
        double beta2 = beta * beta;
        double den = (beta2 + 1.0) * tp + fp + beta2 * fn;
        if (den != 0)
            return (beta2 + 1.0) * tp / den;
        else
            return 1.0;
    }


    /**
     * Get the F1 value.
     *
     * @return The F1 value.
     */
    public double f1() {
        return f(1.0);
    }

    /**
     * Get the accuracy of classification task.
     *
     * @return The accuracy of classification task.
     */
    public double accuracy() {
        double den = tp + tn + fp + fn;
        if (den != 0)
            return (tp + tn) / den;
        else
            return 1.0;
    }

    /**
     * Get the error obtained in classification task.
     *
     * @return The error obtained in classification task.
     */
    public double error() {
        return 1.0 - accuracy();
    }


    /**
     * Get the specificity value.
     *
     * @return The specificity value.
     */
    public double specificity() {
        double den = tn() + fp();
        if (den != 0)
            return tn() / den;
        else
            return 0;
    }


    /**
     * Get the ROC value.
     *
     * @return The ROC value.
     */
    public double roc() {
        return (specificity() + recall()) / 2;
    }


    @Override
    /**
     * Return a string containing all main measures of this contingency table.
     */
    public String toString() {
        StringBuilder sb = new StringBuilder("TP: " + tp() + ", TN: " + tn() + ", FP: " + fp() + ", FN: " + fn() + "\n");
        sb.append("Precision: " + precision() + ", Recall: " + recall() + ", F1:" + f1() + ", Accuracy: " + accuracy() + ", ROC: " + roc());
        return sb.toString();
    }
}
