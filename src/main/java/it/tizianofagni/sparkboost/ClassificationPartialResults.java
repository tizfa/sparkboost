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
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class ClassificationPartialResults implements Serializable {
    public final int partitionID;
    public final String results;
    public final ContingencyTable ct;

    public ClassificationPartialResults(int partitionID, String results, ContingencyTable ct) {
        this.partitionID = partitionID;
        this.results = results;
        this.ct = ct;
    }
}
