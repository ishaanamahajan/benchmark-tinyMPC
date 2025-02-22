#pragma once
#include "main.h"

// Constants from original code
#define NSTATES 12
#define NINPUTS 4
#define NHORIZON 10

// Simple Matrix template class
template<typename T, int ROWS, int COLS>
struct Matrix {
    T data[ROWS][COLS];

    void setZero() {
        memset(data, 0, sizeof(data));
    }

    void setRandom() {
        for(int i = 0; i < ROWS; i++) {
            for(int j = 0; j < COLS; j++) {
                data[i][j] = (float)rand() / RAND_MAX;
            }
        }
    }

    T* operator[](int i) { return data[i]; }
    const T* operator[](int i) const { return data[i]; }

    Matrix<T, ROWS, 1>& col(int j) {
        return *(Matrix<T, ROWS, 1>*)(&data[0][j]);
    }
};