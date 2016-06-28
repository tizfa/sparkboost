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

import org.apache.spark.AccumulableParam;

import java.util.ArrayList;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class DMPartialResultAccumulableParam implements AccumulableParam<ArrayList<SingleDMUpdate>, DMPartialResult> {

    @Override
    public ArrayList<SingleDMUpdate> addAccumulator(ArrayList<SingleDMUpdate> singleDMUpdates, DMPartialResult r) {
        for (int labelID = 0; labelID < r.labelsRes.length; labelID++) {
            singleDMUpdates.add(new SingleDMUpdate(r.docID, labelID, r.labelsRes[labelID]));
        }
        return singleDMUpdates;
    }

    @Override
    public ArrayList<SingleDMUpdate> addInPlace(ArrayList<SingleDMUpdate> r1, ArrayList<SingleDMUpdate> r2) {
        r1.addAll(r2);
        return r1;
    }

    @Override
    public ArrayList<SingleDMUpdate> zero(ArrayList<SingleDMUpdate> initialValue) {
        return new ArrayList<>();
    }
}
