# This code loads the MOROCO data set into memory. It is provided for convenience.
# The data set can be downloaded from <https://github.com/butnaruandrei/MOROCO>.
#
# Copyright (C) 2018  Andrei M. Butnaru, Radu Tudor Ionescu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from os import listdir, makedirs
from os.path import isfile, join, splitext, exists


def loadMOROCODataSamples(subsetName, data_prefix):

    inputSamplesFilePath = (data_prefix + "%s/samples.txt") % (subsetName)
    inputDialectLabelsFilePath = (data_prefix + "%s/dialect_labels.txt") % (subsetName)
    inputCategoryLabelsFilePath = (data_prefix + "%s/category_labels.txt") % (subsetName)
    
    IDs = []
    samples = []
    dialectLabels = []
    categoryLabels = []
    
    # Loading the data samples
    inputSamplesFile = open(inputSamplesFilePath, 'r')
    sampleRows = inputSamplesFile.readlines()
    inputSamplesFile.close()

    for row in sampleRows:
        components = row.split("\t")
        IDs += [components[0]]
        samples += [" ".join(components[1:])]

    # Loading the dialect labels
    inputDialectLabelsFile = open(inputDialectLabelsFilePath, 'r')
    dialectRows = inputDialectLabelsFile.readlines()
    inputDialectLabelsFile.close()
    
    for row in dialectRows:
        components = row.split("\t")
        dialectLabels += [int(components[1])]
    
    # Loading the category labels
    inputCategoryLabelsFile = open(inputCategoryLabelsFilePath, 'r')
    categoryRows = inputCategoryLabelsFile.readlines()
    inputCategoryLabelsFile.close()
    
    for row in categoryRows:
        components = row.split("\t")
        categoryLabels += [int(components[1])]

    # IDs[i] is the ID of the sample samples[i] with the dialect label dialectLabels[i] and the category label categoryLabels[i]
    return IDs, samples, dialectLabels, categoryLabels


def loadMOROCODataSet(data_prefix):
    
    trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels = loadMOROCODataSamples("train", data_prefix)
    print("Loaded %d training samples..." % len(trainSamples))

    validationIDs, validationSamples, validationDialectLabels, validationCategoryLabels = loadMOROCODataSamples("validation", data_prefix)
    print("Loaded %d validation samples..." % len(validationSamples))

    testIDs, testSamples, testDialectLabels, testCategoryLabels = loadMOROCODataSamples("test", data_prefix)
    print("Loaded %d test samples..." % len(testSamples))

    return {
        "train": (trainIDs, trainSamples, trainDialectLabels, trainCategoryLabels),
        "validation": (validationIDs, validationSamples, validationDialectLabels, validationCategoryLabels),
        "test": (testIDs, testSamples, testDialectLabels, testCategoryLabels),
    }


if __name__ == "__main__":
    loadMOROCODataSet("./MOROCO/preprocessed/")
