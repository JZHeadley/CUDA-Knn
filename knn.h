float computeAccuracy(int* confusionMatrix, ArffData* dataset);
int* computeConfusionMatrix(int* predictions, ArffData* dataset);
double euclideanDistance(ArffInstance* instance1, ArffInstance* instance2, int numAttributes);
