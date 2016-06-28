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
 * A single update value in the distribution matrix after the end of an iteration.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class SingleDMUpdate implements Serializable {
    private final int docID;
    private final int labelID;
    private final double result;

    public SingleDMUpdate(int docID, int labelID, double result) {
        this.docID = docID;
        this.labelID = labelID;
        this.result = result;
    }

    public int getDocID() {
        return docID;
    }

    public int getLabelID() {
        return labelID;
    }

    public double getResult() {
        return result;
    }

}
