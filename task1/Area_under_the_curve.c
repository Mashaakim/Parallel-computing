#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <float.h>

double f(double x) {
    return 4/(1+x*x);
}

int main(int argc, char *argv[]) {
    double a, b;
    int i;
    int N = atoi(argv[1]);
    int myrank, p;
    double sum, part_sum;
    double h = 1.0/N;
    double eps = 10e-12;

    MPI_Status Status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    double parts[p];
    double splitting[p+1];
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        double mainsum = 0;
        double x = 0;

        for (i = 0; i < N; i++) {
            mainsum += 0.5*(f(x)+f(x+h))*h;
            x += h;
        }
        printf("Sum by the main process is %f\n", mainsum);
        double T_begin = MPI_Wtime();

        splitting[0] = 0;
        splitting[1] = h*(N/p+N%p);
        splitting[p] = 1;

        for (i = 2; i < p; i++){
            splitting[i] = splitting[i-1] + h*(N/p);
        }

        for (i = 1; i < p; i++)
        {
            MPI_Send(&splitting[i], 2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }

        part_sum = 0;
        a = splitting[myrank];
        b = splitting[myrank+1];
        while (a + h <= b + eps) {
            part_sum += 0.5*(f(a)+f(a+h))*h;
            a+=h;
        }
        printf("I am the %d and my sum is %f\n", myrank, part_sum);
        parts[myrank] = part_sum;
        if (p == 1)
        {
            double T_total = MPI_Wtime() - T_begin;
            MPI_Finalize();
            printf("Time on one thread is %.5f\n", T_total);
            return 0;
        }
        sum = parts[myrank];
        for (i=1; i<p; i++)
        {
            MPI_Recv(&parts[i], 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &Status);
            sum += parts[i];
        }

        printf("Total sum is %f\n", sum);
        double T_total = MPI_Wtime() - T_begin;
        printf("Time on %d threads is %.5f\n", p, T_total);
        MPI_Finalize();
    }

    if (myrank != 0)
    {
        MPI_Recv(&splitting[myrank], 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &Status);

        part_sum = 0;
        a = splitting[myrank]; //границы интегрирования
        b = splitting[myrank+1];

        while(a + h <= b + eps) {
            part_sum += 0.5*(f(a)+f(a+h))*h;
            a+=h;
        }
        parts[myrank] = part_sum;
        printf("I am the %d and my sum is %f\n", myrank, part_sum);
        MPI_Send(&parts[myrank], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Finalize();
    }
    return 0;
}