﻿


#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <mpi.h>

#define MIN(a,b)  ((a)<(b)?(a):(b))

int main(int argc, char *argv[])
{
	int    count;        /* Local prime count */
	double elapsed_time; /* Parallel execution time */
	int    first;        /* Index of first multiple */
	int    global_count; /* Global prime count */
	int    high_value;   /* Highest value on this proc */
	int    i;
	int    id;           /* Process ID number */
	int    index;        /* Index of current prime */
	int    low_value;    /* Lowest value on this proc */
	char  *marked;       /* Portion of 2,...,'n' */
	int    n;            /* Sieving from 2, ..., 'n' */
	int    p;            /* Number of processes */
	int    proc0_size;   /* Size of proc 0's subarray */
	int    prime;        /* Current prime */
	int    size;         /* Elements in 'marked' */

	MPI_Init(&argc, &argv);

	/* Start the timer */

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed_time = -MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);

	if (argc != 2) {
		if (!id) printf("Command line: %s <m>\n", argv[0]);
		MPI_Finalize();
		exit(1);
	}

	n = atoi(argv[1]);

	//分块存在一定问题
	/*low_value = 2 + id * (n - 1) / p;
	high_value = 1 + (id + 1)*(n - 1) / p;
	size = high_value - low_value + 1;*/

	int q = (n + 1) / p, r = (n + 1) % p;
	if (id < r) {
		low_value = id * (q + 1);
		high_value = low_value + (q + 1) - 1;
	}
	else {
		low_value = r * (q + 1) + (id - r) * q;
		high_value = low_value + q - 1;
	}
	if (id == 0) low_value = 3;
	size = (high_value)/2 - low_value/2;


	proc0_size = (n - 1) / p;

	if ((2 + proc0_size) < (int)sqrt((double)n)) {
		if (!id) printf("Too many processes\n");
		MPI_Finalize();
		exit(1);
	}

	/* Allocate this process's share of the array. */

	marked = (char *)malloc(size);

	if (marked == NULL) {
		printf("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit(1);
	}

	for (i = 0; i < size; i++) marked[i] = 0;
	if (!id) index = 0;
	prime = 3;
	do {
		if (prime * prime > low_value)
			first = prime * prime - low_value;
		else {
			if (!(low_value % prime)) first = 0;
			else first = prime - (low_value % prime);
		}
		if ((first + low_value) % 2 == 0)first += prime;
		for (i = first>>1; i < size; i += prime) marked[i] = 1;
		if (!id) {
			while (marked[++index]);
			prime = index * 2 + 3;
		}
		if (p > 1) MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} while (prime * prime <= n);
	count = 0;
	for (i = 0; i < size; i++)
		if (!marked[i]) count++;
	if (p > 1) MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM,
		0, MPI_COMM_WORLD);

	/* Stop the timer */

	elapsed_time += MPI_Wtime();


	/* Print the results */

	if (!id) {
		printf("There are %d primes less than or equal to %d\n",
			global_count + 1, n);
		printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
	}
	MPI_Finalize();
	return 0;
}
