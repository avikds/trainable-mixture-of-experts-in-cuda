"""
Trainable Mixture of Experts in CUDA scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

/*
 * scaffold.cu - Drives a small Mixture-of-Experts training run end-to-end.
 * Allocates tiny toy input/target, calls moe_forward / moe_backward via
 * moe_training_loop, and prints loss history and a few output values.
 */

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while(0)

static float frand() { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

int main() {
    srand(0);

    // Toy hyper-parameters
    const int T = 8;     // tokens
    const int D = 4;     // input dim
    const int H = 6;     // hidden dim
    const int O = 4;     // output dim
    const int E = 4;     // num experts
    const int K = 2;     // top-k
    const int total_slots = T * K;
    const float lr = 0.05f;
    const float aux_scale = 0.01f;
    const int num_steps = 10;

    // Host buffers for input/target/parameters
    std::vector<float> hX(T * D), hY(T * O);
    std::vector<float> hWg(D * E), hWup(E * D * H), hBup(E * H), hWdn(E * H * O), hBdn(E * O);
    for (auto& v : hX)  v = frand();
    for (auto& v : hY)  v = frand();
    for (auto& v : hWg) v = frand() * 0.1f;
    for (auto& v : hWup) v = frand() * 0.1f;
    for (auto& v : hBup) v = 0.0f;
    for (auto& v : hWdn) v = frand() * 0.1f;
    for (auto& v : hBdn) v = 0.0f;

    // Device parameter buffers
    float *dX, *dTgt;
    float *dWg, *dWup, *dBup, *dWdn, *dBdn;
    CUDA_CHECK(cudaMalloc(&dX,   T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTgt, T * O * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWg,  D * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWup, E * D * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBup, E * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWdn, E * H * O * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBdn, E * O * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,   hX.data(),   T * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTgt, hY.data(),   T * O * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWg,  hWg.data(),  D * E * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWup, hWup.data(), E * D * H * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBup, hBup.data(), E * H * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWdn, hWdn.data(), E * H * O * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBdn, hBdn.data(), E * O * sizeof(float), cudaMemcpyHostToDevice));

    // Forward-pass scratch
    float *dRouterLogits, *dRouterProbs, *dTopkVals, *dTopkGates;
    int   *dTopkIdx, *dCounts, *dOffsets, *dSlotK, *dSlotTok;
    float *dGathered, *dHpre, *dHpost, *dExpOut, *dOut;
    CUDA_CHECK(cudaMalloc(&dRouterLogits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRouterProbs,  T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopkVals,     T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopkGates,    T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTopkIdx,      T * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCounts,       E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOffsets,      E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSlotK,        total_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSlotTok,      total_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dGathered,     total_slots * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHpre,         total_slots * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHpost,        total_slots * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dExpOut,       total_slots * O * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut,          T * O * sizeof(float)));

    // Backward scratch
    float *dGradOut, *dGradExpOut, *dGradGathered, *dGradHpost, *dGradHpre;
    float *dGradTopkGates, *dGradTopkVals, *dGradRouterProbs, *dGradRouterLogits;
    float *dGradIn, *dGradWg, *dGradWup, *dGradBup, *dGradWdn, *dGradBdn;
    CUDA_CHECK(cudaMalloc(&dGradOut,          T * O * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradExpOut,       total_slots * O * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradGathered,     total_slots * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradHpost,        total_slots * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradHpre,         total_slots * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradTopkGates,    T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradTopkVals,     T * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradRouterProbs,  T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradRouterLogits, T * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradIn,           T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradWg,           D * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradWup,          E * D * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradBup,          E * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradWdn,          E * H * O * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGradBdn,          E * O * sizeof(float)));

    // Run one forward to view initial output
    moe_forward(dX, dWg, dWup, dBup, dWdn, dBdn,
                dRouterLogits, dRouterProbs, dTopkVals, dTopkIdx, dTopkGates,
                dCounts, dOffsets, dSlotK, dSlotTok,
                dGathered, dHpre, dHpost, dExpOut, dOut,
                T, D, H, O, E, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hOut(T * O);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, T * O * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Initial output[0]: ");
    for (int j = 0; j < O; ++j) printf("%.4f ", hOut[j]);
    printf("\nTarget[0]:        ");
    for (int j = 0; j < O; ++j) printf("%.4f ", hY[j]);
    printf("\n");

    // Train loop: run num_steps training steps and collect loss history.
    std::vector<float> hLossHistory(num_steps, 0.0f);
    for (int step = 0; step < num_steps; ++step) {
        float h_loss = 0.0f;
        // Call training step directly to drive forward+backward+SGD update.
        // ctx-style fields are not available, so we drive the steps explicitly:
        moe_forward(dX, dWg, dWup, dBup, dWdn, dBdn,
                    dRouterLogits, dRouterProbs, dTopkVals, dTopkIdx, dTopkGates,
                    dCounts, dOffsets, dSlotK, dSlotTok,
                    dGathered, dHpre, dHpost, dExpOut, dOut,
                    T, D, H, O, E, K);

        float *dLoss;
        CUDA_CHECK(cudaMalloc(&dLoss, sizeof(float)));
        mse_loss_forward(dOut, dTgt, dLoss, T, O);
        CUDA_CHECK(cudaMemcpy(&h_loss, dLoss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dLoss));
        hLossHistory[step] = h_loss;

        mse_loss_backward(dOut, dTgt, dGradOut, T, O);
        zero_buffer(dGradIn, T * D);
        zero_buffer(dGradWg, D * E);
        zero_buffer(dGradWup, E * D * H);
        zero_buffer(dGradBup, E * H);
        zero_buffer(dGradWdn, E * H * O);
        zero_buffer(dGradBdn, E * O);

        moe_backward(dX, dWg, dWup, dBup, dWdn, dBdn,
                     dRouterLogits, dRouterProbs, dTopkVals, dTopkIdx, dTopkGates,
                     dCounts, dOffsets, dSlotK, dSlotTok,
                     dGathered, dHpre, dHpost, dExpOut,
                     dGradOut, dGradExpOut, dGradGathered, dGradHpost, dGradHpre,
                     dGradTopkGates, dGradTopkVals, dGradRouterProbs, dGradRouterLogits,
                     dGradIn, dGradWg, dGradWup, dGradBup, dGradWdn, dGradBdn,
                     T, D, H, O, E, K, aux_scale);

        sgd_update_parameters(dWg,  dGradWg,  lr, D * E);
        sgd_update_parameters(dWup, dGradWup, lr, E * D * H);
        sgd_update_parameters(dBup, dGradBup, lr, E * H);
        sgd_update_parameters(dWdn, dGradWdn, lr, E * H * O);
        sgd_update_parameters(dBdn, dGradBdn, lr, E * O);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nLoss history:\n");
    for (int s = 0; s < num_steps; ++s) printf("  step %2d  loss=%.6f\n", s, hLossHistory[s]);

    // Final forward to peek at trained output
    moe_forward(dX, dWg, dWup, dBup, dWdn, dBdn,
                dRouterLogits, dRouterProbs, dTopkVals, dTopkIdx, dTopkGates,
                dCounts, dOffsets, dSlotK, dSlotTok,
                dGathered, dHpre, dHpost, dExpOut, dOut,
                T, D, H, O, E, K);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, T * O * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nFinal output[0]: ");
    for (int j = 0; j < O; ++j) printf("%.4f ", hOut[j]);
    printf("\n");

    // Cleanup
    cudaFree(dX); cudaFree(dTgt);
    cudaFree(dWg); cudaFree(dWup); cudaFree(dBup); cudaFree(dWdn); cudaFree(dBdn);
    cudaFree(dRouterLogits); cudaFree(dRouterProbs); cudaFree(dTopkVals); cudaFree(dTopkGates);
    cudaFree(dTopkIdx); cudaFree(dCounts); cudaFree(dOffsets); cudaFree(dSlotK); cudaFree(dSlotTok);
    cudaFree(dGathered); cudaFree(dHpre); cudaFree(dHpost); cudaFree(dExpOut); cudaFree(dOut);
    cudaFree(dGradOut); cudaFree(dGradExpOut); cudaFree(dGradGathered);
    cudaFree(dGradHpost); cudaFree(dGradHpre);
    cudaFree(dGradTopkGates); cudaFree(dGradTopkVals);
    cudaFree(dGradRouterProbs); cudaFree(dGradRouterLogits);
    cudaFree(dGradIn); cudaFree(dGradWg); cudaFree(dGradWup);
    cudaFree(dGradBup); cudaFree(dGradWdn); cudaFree(dGradBdn);
    return 0;
}
