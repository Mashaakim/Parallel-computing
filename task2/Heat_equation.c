#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

double exact_solution(double x) {
    double T = 0.1;
    double L = 1.0;
    double u = (4/3.1415)*exp((-3.14*3.14*T)/(L*L))*sin(3.14*x/L);
    double sum = u;
    int i = 1;
    while(u > 0.00001) {
        u = (4 / 3.14) * (exp(-pow(3.14, 2) * pow(2 * i + 1, 2) * T / pow(L, 2)) / (2 * i + 1)) *
            sin(3.14 * (2 * i + 1) * x / L);
        sum += u;
        i++;
    }
    return sum;
}

int main(int argc, char * argv[])
{
    double T = 0.1;
    int i = 0, j = 0; //итераторы
    int myrank, p;
    int N = atoi(argv[1]);
    double h = 1.0/(N-1); //шаг по пространству
    double dt = 0.0002; //шаг по времени
    double u_0 = 1; //начальное распределение
    double u_1 = 0; //на концах
    int times = T / dt; //число итераций по времени
    int d = 0;

    double T_begin;

    MPI_Status Status;
    MPI_Init(&argc, &argv);
    T_begin = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


    int dn = N / p; //число точек на один процесс
    int start = dn * myrank; //начало работы процесса - для каждого своё
    int finish = start + dn; //конец работы
    int sendsize = 0;

    if (myrank == 0) //для первого
        start = 1;
    if (myrank == p - 1) //для последнего
        finish = N - 1;


    double * u[2];

    //заполняем массив начальными значениями
    for (i = 0; i < 2; i++)
        u[i] = (double *)calloc(N, sizeof(double));
    for (j = 0; j < N; j++)
        for (i = 0; i < 2; i++)
            u[i][j] = u_0;
    u[0][0] = u_1;
    u[1][0] = u_1;
    u[0][N-1] = u_1;
    u[1][N-1] = u_1;

    for (i = 0; i < times; i++)
    {
        if (myrank % 2 == 1) //нечетные процессы
        {
            if (myrank < p - 1)
            {
                MPI_Ssend(&u[d][finish - 1], 1, MPI_DOUBLE, myrank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&u[d][finish], 1, MPI_DOUBLE, myrank + 1, 0, MPI_COMM_WORLD, &Status);
            }
            if (myrank != 0)
            {
                MPI_Recv(&u[d][start - 1], 1, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD, &Status);
                MPI_Ssend(&u[d][start], 1, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD);
            }
        } else //четные процессы
        {
            if (myrank != 0)
            {
                MPI_Recv(&u[d][start - 1], 1, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD, &Status);
                MPI_Ssend(&u[d][start], 1, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD);
            }
            if (myrank < p - 1)
            {
                MPI_Ssend(&u[d][finish - 1], 1, MPI_DOUBLE, myrank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&u[d][finish], 1, MPI_DOUBLE, myrank + 1, 0, MPI_COMM_WORLD, &Status);
            }
        }

        for (j = start; j < finish; j++) //итерация
            u[1-d][j] = u[d][j] + (dt/h*h) * (u[d][j-1] - 2.0 * u[d][j] + u[d][j+1]);
        d = 1-d;
    }

    //после итераций все процессы отравляют нулевому свои результаты
    if (myrank != 0)
    {
        sendsize = dn;
        if (myrank == p - 1)
            sendsize += N % p;
        MPI_Send(u[d] + dn * myrank, sendsize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    //нулевой процесс принимает результаты
    if (myrank == 0)
        for (i = 1; i < p; i++) {
            sendsize = dn;
            if (i == p - 1)
                sendsize += N % p;
            MPI_Recv(u[d] + dn * i, sendsize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);
        }

    //один процесс выводит все результаты
    if (myrank == 0)
    {
        for (i = 0; i < 11; i++)
        {
            printf("U[%.1lg] = %lg\n", 0.1 * i, u[d][i]);
            if (i*0.1 != 1)
                printf("Exact U[%.1lg] = %lg\n", 0.1 * i, exact_solution(0.1*i));
            else
                printf("Exact U[%.1lg] = %d\n", 0.1 * i, 0);
        }
        printf("Number of processes: %d, Total time: %lf\n", p, MPI_Wtime() - T_begin);
    }

    //освобождаем память
    for (i = 0; i < 2; i++)
        free(u[i]);

    MPI_Finalize();

    return 0;
}