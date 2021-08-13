#include "nanoflann.hpp"
using namespace nanoflann;

#include "KDTreeVectorOfVectorsAdaptor.h"

#include <ctime>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>

const int SAMPLES_DIM = 15;

//typedef std::vector<std::vector<double> > my_vector_of_vectors_t;
template<int Dimension>
using myVectors = std::vector<Eigen::Matrix<double, Dimension, 1>>;

template<int Dimension>
void generateRandomPointCloud(myVectors<Dimension> &samples, const size_t N, const size_t dim, const double max_range = 10.0)
{
	std::cout << "Generating " << N << " random points...";
	samples.resize(N);
	for (size_t i = 0; i < N; i++)
	{
		for (size_t d = 0; d < dim; d++)
			samples[i][d] = max_range * (rand() % 1000) / (1000.0);
	}
	std::cout << "done\n";
}

void kdtree_demo(const size_t nSamples, const size_t dim)
{
	myVectors<SAMPLES_DIM> samples;

	const double max_range = 20;

	// Generate points:
	generateRandomPointCloud(samples, nSamples, dim, max_range);

	// Query point:
	//std::vector<double> query_pt(dim);
	//for (size_t d = 0; d < dim; d++)
	//	query_pt[d] = max_range * (rand() % 1000) / (1000.0);
	Eigen::Matrix<double, SAMPLES_DIM, 1> query_pt;
	for( size_t d = 0; d < dim; d ++ ){
		query_pt[d] = max_range * (rand() % 1000) / (1000.0);
	}

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------
	typedef KDTreeVectorOfVectorsAdaptor< myVectors<SAMPLES_DIM>, double >  my_kd_tree_t;

	my_kd_tree_t   mat_index(dim, samples, 10);
	mat_index.index->buildIndex();

	// do a knn search
	const size_t num_results = 3;
	std::vector<size_t>   ret_indexes(num_results);
	std::vector<double> out_dists_sqr(num_results);

	nanoflann::KNNResultSet<double> resultSet(num_results);

	resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
	mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

	std::cout << "knnSearch(nn=" << num_results << "): \n";
	for (size_t i = 0; i < num_results; i++)
		std::cout << "ret_index[" << i << "]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << std::endl;


}

int main()
{
	// Randomize Seed
	srand(static_cast<unsigned int>(time(nullptr)));
	kdtree_demo(1000 /* samples */, SAMPLES_DIM /* dim */);
	

	return 0;
}
