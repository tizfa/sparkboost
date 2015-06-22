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



    public int tp() {
        return tp;
    }

    public int tn() {
        return tn;
    }

    public int fp() {
        return fp;
    }

    public int fn() {
        return fn;
    }

    public int total() {
        return fn + fp + tn + tp;
    }

    public double precision() {
        double den = tp + fp;
        if (den != 0)
            return tp / den;
        else
            return 1.0;
    }

    public double recall() {
        double den = tp + fn;
        if (den != 0)
            return tp / den;
        else
            return 1.0;
    }

    public double f(double beta) {
        double beta2 = beta * beta;
        double den = (beta2 + 1.0) * tp + fp + beta2 * fn;
        if (den != 0)
            return (beta2 + 1.0) * tp / den;
        else
            return 1.0;
    }


    public double f1() {
        return f(1.0);
    }

    public double accuracy() {
        double den = tp + tn + fp + fn;
        if (den != 0)
            return (tp + tn) / den;
        else
            return 1.0;
    }

    public double error() {
        return 1.0 - accuracy();
    }

    public double pd() {
        double den = tp + tn + fp + fn;
        if (den == 0)
            return 0.0;
        return Math.abs((fp - fn) / den);
    }

    public double relativePd() {
        double den = tp + tn + fp + fn;
        if (den == 0)
            return 0;

        double pd = pd();
        double pos = (tp + fn) / den;

        if (pos == 0) {
            if (pd == 0)
                return 0;
            else
                return 1.0;
        } else
            return Math.min(pd / pos, 1.0);
    }



    public double specificity() {
        double den = tn() + fp();
        if (den != 0)
            return tn() / den;
        else
            return 0;
    }

    public double fpr() {
        return 1.0 - specificity();
    }

    public double roc() {
        return (specificity() + recall()) / 2;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("TP: "+tp()+", TN: "+tn()+", FP: "+fp()+", FN: "+fn()+"\n");
        sb.append("Precision: "+precision()+", Recall: "+recall()+", F1:"+f1()+", Accuracy: "+accuracy()+", ROC: "+roc());
        return sb.toString();
    }
}
