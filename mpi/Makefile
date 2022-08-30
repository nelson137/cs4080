SHELL:=/bin/bash

all: prep mpi_test pi_estimator word_counter

prep:
	. /usr/share/Modules/init/bash; \
        module add openmpi/openmpi-3.1.4; \
        module add boost/boost-1.74.0

parallel = ${MPI_THREADS}

run: run_opening run_test run_pi run_counter
	@echo " "; echo "Runs completed"

run_opening:
	@echo "Running tests with ${parallel} threads";

run_test: mpi_test
	@echo "Running the simple mpi test" ;\
	time mpirun -np ${parallel} mpi_test ;

run_pi: pi_estimator
	@echo " "; echo "Running PI estimator" ; \
	time mpirun -np ${parallel} pi_estimator ;

run_counter: word_counter
	@echo " "; echo "Running Word Counter" ; \
	time mpirun -np ${parallel} ./word_counter genesis.txt

# Test Program
mpi_test: mpi_test.o
	mpic++ -o mpi_test mpi_test.o

mpi_test.o:
	mpic++ -c mpi_test.cpp


# PI estimator
pi_estimator: mpi_pi_reduce.o dboard.o
	mpicc -o pi_estimator mpi_pi_reduce.o dboard.o

mpi_pi_reduce.o: mpi_pi_reduce.c
	mpicc -c mpi_pi_reduce.c

dboard.o: dboard.c
	mpicc -c dboard.c

# Book Counter

word_counter: sample_file_word_count.cpp
	mpic++ -std=gnu++11 -g -o word_counter sample_file_word_count.cpp

clean:
	rm -f mpi_test pi_estimator word_counter *.o 

