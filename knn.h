#include "libarff/arff_data.h"
int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
	int* confusionMatrix = (int*) calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses

	for (int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
	{
		int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
		int predictedClass = predictions[i];

		confusionMatrix[trueClass * dataset->num_classes() + predictedClass]++;
	}

	return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
	int successfulPredictions = 0;

	for (int i = 0; i < dataset->num_classes(); i++)
	{
		successfulPredictions += confusionMatrix[i * dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
	}

	return successfulPredictions / (float) dataset->num_instances();
}
