/*
 ============================================================================
 Name        : KNN.cu
 Author      : jzheadley
 Version     :
 Copyright   :
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <limits.h>

#include "libarff/arff_parser.h"

#include "knn.h"
using namespace std;
#define K 3

#define NUM_STREAMS 2

__global__ void computeDistances(int numInstances, int numAttributes, float* dataset, float* distances)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int row = tid / numInstances; // instance1Index
	int column = tid - ((tid / numInstances) * numInstances); //instance2Index
	if ((tid < numInstances * numInstances))
	{
		float sum = 0;
		int instance1 = row * numAttributes;
		int instance2 = column * numAttributes;
		for (int atIdx = 0; atIdx < numAttributes - 1; atIdx++) // numAttributes -1 since we don't want to compare class in the distance because that doesn't make sense
		{
			sum += ((dataset[instance1 + atIdx] - dataset[instance2 + atIdx]) * (dataset[instance1 + atIdx] - dataset[instance2 + atIdx]));
		}
		distances[row * numInstances + column] = (float) sqrt(sum);
		distances[column * numInstances + row] = distances[row * numInstances + column]; //set the distance for the other half of the pair we just computed
	}
}

__inline__ __device__ void reduceToK(float* distancesTo, int* indexes, int k, int curSize)
{
	// we're just going to do a simple bubble sort and pretend the elements past k don't exist
	float tmp;
	unsigned char idx;
	for (int i = 0; i < curSize - 1; i++)
	{
		for (int j = 0; j < curSize - i - 1; j++)
		{
			if (distancesTo[j] > distancesTo[j + 1])
			{
				tmp = distancesTo[j];
				idx = indexes[j];
				distancesTo[j] = distancesTo[j + 1];
				indexes[j] = indexes[j + 1];
				distancesTo[j + 1] = tmp;
				indexes[j + 1] = idx;
			}
		}
	}
}
__inline__ __device__ int vote(float* distancesTo, int *indexes, float* dataset, int k, int numAttributes)
{
	int classVotes[32]; // can technically parallelize this reading in the class num and probably should come back and do that
	bool duplicate = false;
	int finalClass;
	int mostVotes = 0;
	for (int i = 0; i < 32; i++)
		classVotes[i] = 0;
	for (int i = 0; i < k; i++)
	{
		int classNum = dataset[indexes[i] * numAttributes + numAttributes - 1];
		classVotes[classNum] += 1;
	}
	for (int i = 0; i < 32; i++) // have to find highest count first
	{
		if (classVotes[i] > mostVotes)
		{
			finalClass = i;
			mostVotes = classVotes[i];
		}
	}
	for (int i = 0; i < 32; i++) // then compare to that to ensure we don't have duplicates
	{
		if (classVotes[i] == mostVotes && classVotes[i] > 0)
			duplicate = true;
	}
	if (duplicate)
	{
		if ((k - 1) > 0) // I'm not quite sure why I'm detecting dupes when k=1 but I am soo... this takes care of that and makes everything correct again...
		{
			return vote(distancesTo, indexes, dataset, k - 1, numAttributes);
		}
	}
	return finalClass;
}

__global__ void knn(int* predictions, float*distances, float*dataset, int numAttributes)
{
	__shared__ int indexes[256];
	__shared__ float distancesTo[256];
	// gridDim.x is numInstances
	int bestInstanceId;
	float bestDistance = INT_MAX;
	int instanceFrom = blockIdx.x * gridDim.x;
	int distancePos;
	int rowBoundary = instanceFrom + gridDim.x - 1;
	if (blockDim.x < gridDim.x)
	{ //If we have more elements than threads we need to do an inital reduction to fit into our shared mem
		if (threadIdx.x < blockDim.x) // only want 256 threads to come into this otherwise we will go out of bounds of our shared mem
		{
			for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) // will try to make this more coalesced later
			{
				if (i == blockIdx.x) // don't need to include the diagonal
					continue;

				distancePos = instanceFrom + i;
				if (distancePos > rowBoundary)
				{ // should take care of the final elements
					break;
				}
				if (distances[distancePos] < bestDistance)
				{
					bestDistance = distances[distancePos];
					bestInstanceId = i;
				}
			}
			indexes[threadIdx.x] = bestInstanceId;
			distancesTo[threadIdx.x] = bestDistance;
		}
		__syncthreads();

		if (threadIdx.x < blockDim.x / 2) // only need the first half(128) of the threads to work on the 256 length shared mem arrays
		{
			int s;
			// this for should probably have the conditional of (s>>1) > k but if I do that I don't reduce enough sooo...
			// we're going with this until I find that error and just upping s back up after this for
			for (s = blockDim.x / 2; (s) > K; s >>= 1)
			{
				if (threadIdx.x < s)
				{
					if (distancesTo[threadIdx.x + s] < distancesTo[threadIdx.x])
					{
						distancesTo[threadIdx.x] = distancesTo[threadIdx.x + s];
						indexes[threadIdx.x] = indexes[threadIdx.x + s];
					}
					__syncthreads();
				}
			}
			s *= 2;
			__syncthreads();
			if (s > K && threadIdx.x == 1)
			{ // we need to reduce it just a little more
			  // remember to change both the indexes and distancesTo arrays
				reduceToK(distancesTo, indexes, K, s);
			}
			__syncthreads();
			if (threadIdx.x == 1)
				predictions[blockIdx.x] = vote(distancesTo, indexes, dataset, K, numAttributes);
		}
	}
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		if (argc != 3)
		{
			cout << "Usage: ./main datasets/datasetFile.arff" << endl;
			exit(0);
		}
	}

	ArffParser parser(argv[1]);

	ArffData* dataset = parser.parse();

	cudaStream_t *streams = (cudaStream_t*) malloc(NUM_STREAMS * sizeof(cudaStream_t));
	for (int i = 0; i < NUM_STREAMS; i++) // multiple streams
		cudaStreamCreate(&streams[i]);

	int numInstances = dataset->num_instances();
	int numAttributes = dataset->num_attributes();
	int* h_predictions = (int *) calloc(numInstances, sizeof(int));
	printf("We're classifying %i instances with %i attributes\n", numInstances, numAttributes);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	int numTriangularSpaces = (numInstances * numInstances); //(numInstances * (numInstances - 1)) / 2; // don't actually need the diagonal since its all 0's so we can have numInstances-1 instead of + 1 but math is hard
	float* h_dataset, *h_distances;
	cudaMallocHost(&h_dataset, sizeof(float) * numInstances * numAttributes);
	cudaMallocHost(&h_distances, sizeof(float) * numTriangularSpaces);
	printf("numTriangularSpaces is %i\n", numTriangularSpaces);

	for (int instanceNum = 0; instanceNum < numInstances; instanceNum++)
	{
		// each 'row' will be an instances
		// each 'column' a specific attribute
		ArffInstance* instance = dataset->get_instance(instanceNum);
		for (int attributeNum = 0; attributeNum < numAttributes; attributeNum++)
		{
			h_dataset[instanceNum * numAttributes + attributeNum] = (float) instance->get(attributeNum)->operator int32();
		}

	}

	float* d_dataset;
	float* d_distances;
	int* d_predictions;

	cudaMalloc(&d_predictions, numInstances * sizeof(int));
	cudaMalloc(&d_dataset, numInstances * numAttributes * sizeof(float));
	cudaMallocHost(&d_distances, numTriangularSpaces * sizeof(float));

	int threadsPerBlock = 256;
//	int blocksPerGrid = (numInstances + threadsPerBlock - 1) / threadsPerBlock;
	int blocksPerGrid = ((numInstances * numInstances) + threadsPerBlock - 1) / threadsPerBlock;
	cudaEventRecord(start);

	cudaMemcpyAsync(d_dataset, h_dataset, numInstances * numAttributes * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(d_distances, h_distances, numTriangularSpaces * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
	computeDistances<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(numInstances, numAttributes, d_dataset, d_distances);

	cudaMemcpyAsync(d_predictions, h_predictions, numInstances * sizeof(int), cudaMemcpyHostToDevice, streams[1]);
	cudaStreamSynchronize(streams[0]); // need this to ensure that the previous kernel computing the distances is finished otherwise we might not have the full distance matrix
	knn<<<numInstances, 256, 0, streams[1]>>>(d_predictions, d_distances, d_dataset, numAttributes);
	cudaMemcpyAsync(h_predictions, d_predictions, numInstances * sizeof(int), cudaMemcpyDeviceToHost, streams[1]);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaError_t cudaError = cudaGetLastError();

	if (cudaError != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	int* confusionMatrix = computeConfusionMatrix(h_predictions, dataset);
	float accuracy = computeAccuracy(confusionMatrix, dataset);
	printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", numInstances, (long long unsigned int) milliseconds,
			accuracy);

	return 0;
}
