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

import org.apache.spark.AccumulableParam;

import java.util.ArrayList;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class DMNormalizationAccumulableParam implements AccumulableParam<ArrayList<Double>, DMPartialResult> {

    @Override
    public ArrayList<Double> addAccumulator(ArrayList<Double> normalization, DMPartialResult r) {
        for (int labelID = 0; labelID < r.labelsRes.length; labelID++) {
            normalization.set(labelID, normalization.get(labelID) + r.labelsRes[labelID]);
        }
        return normalization;
    }

    @Override
    public ArrayList<Double> addInPlace(ArrayList<Double> r1, ArrayList<Double> r2) {
        for (int labelID = 0; labelID < r1.size(); labelID++) {
            r1.set(labelID, r1.get(labelID) + r2.get(labelID));
        }
        return r1;
    }

    @Override
    public ArrayList<Double> zero(ArrayList<Double> initialValue) {
        return initialValue;
    }
}
