"""
Trainable Mixture of Experts in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_naive_kernel
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread computes exactly one element of C.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against threads outside the output matrix.
    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    // Compute the dot product of A's row and B's column.
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

# Step 2 - matmul_tiled_kernel
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared-memory tiles for A and B.
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    // Global row and column for the output element computed by this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Process the K dimension one tile at a time.
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        int aCol = tile * 16 + threadIdx.x;
        int bRow = tile * 16 + threadIdx.y;

        // Cooperatively load A and B into shared memory.
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Make sure the entire tile has been loaded before using it.
        __syncthreads();

        // Compute the partial dot product for this tile.
        for (int k = 0; k < 16; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Make sure all threads have finished using the current tiles
        // before the shared memory is overwritten in the next iteration.
        __syncthreads();
    }

    // Write the result, guarding edge threads for non-multiple dimensions.
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

# Step 3 - matmul_at_b_kernel
__global__ void matmul_at_b_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread computes exactly one element C[m, n].
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against threads outside the output matrix.
    if (m >= M || n >= N) {
        return;
    }

    float sum = 0.0f;

    // A is stored as K x M, so A^T[m, k] = A[k, m].
    // B is stored as K x N, so B[k, n] = B[k * N + n].
    for (int k = 0; k < K; ++k) {
        sum += A[k * M + m] * B[k * N + n];
    }

    C[m * N + n] = sum;
}

# Step 4 - matmul_a_bt_kernel
__global__ void matmul_a_bt_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread computes one element C[i, j].
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against threads outside the output matrix.
    if (i >= M || j >= N) {
        return;
    }

    float sum = 0.0f;

    // B is stored as N x K, so B^T[k, j] = B[j, k].
    for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[j * K + k];
    }

    C[i * N + j] = sum;
}

# Step 5 - add_bias_row_kernel
__global__ void add_bias_row_kernel(float* Y, const float* bias, int M, int N) {
    // Map each thread to one element Y[i, j].
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against threads outside the matrix.
    if (i >= M || j >= N) {
        return;
    }

    // Add the corresponding bias value to this column.
    Y[i * N + j] += bias[j];
}

# Step 6 - reduce_rows_to_bias_grad_kernel
__global__ void reduce_rows_to_bias_grad_kernel(const float* dY, float* dbias, int M, int N) {
    // One thread handles one column.
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against threads outside the column range.
    if (j >= N) {
        return;
    }

    float sum = 0.0f;

    // Sum the gradient over all rows for this column.
    for (int i = 0; i < M; ++i) {
        sum += dY[i * N + j];
    }

    dbias[j] = sum;
}

# Step 7 - elementwise_add_kernel
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int n) {
    // Grid-stride loop so each thread can process multiple elements.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

# Step 8 - relu_forward_kernel
__global__ void relu_forward_kernel(const float* x, float* y, int n) {
    // Each thread handles one element.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-range threads.
    if (i >= n) {
        return;
    }

    // ReLU: y[i] = max(0, x[i]).
    y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

# Step 9 - relu_backward_kernel
__global__ void relu_backward_kernel(const float* x, const float* dy, float* dx, int n) {
    // Each thread handles one element.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-range threads.
    if (i >= n) {
        return;
    }

    // ReLU derivative: 1 for x[i] > 0, otherwise 0.
    dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
}

# Step 10 - gelu_forward_kernel
__global__ void gelu_forward_kernel(const float* x, float* y, int n) {
    // Use a grid-stride loop so each thread can process one or more elements.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_COEFF = 0.044715f;

    for (int i = idx; i < n; i += stride) {
        float value = x[i];
        float value_cubed = value * value * value;

        float inner =
            SQRT_2_OVER_PI * (value + GELU_COEFF * value_cubed);

        y[i] = 0.5f * value * (1.0f + tanhf(inner));
    }
}

# Step 11 - gelu_backward_kernel
__global__ void gelu_backward_kernel(const float* x, const float* dy, float* dx, int n) {
    // Grid-stride loop so all elements are processed.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_COEFF = 0.044715f;

    for (int i = idx; i < n; i += stride) {
        float value = x[i];

        // u = sqrt(2/pi) * (x + 0.044715 * x^3)
        float x_squared = value * value;
        float x_cubed = x_squared * value;
        float u = SQRT_2_OVER_PI * (value + GELU_COEFF * x_cubed);

        // tanh(u)
        float tanh_u = tanhf(u);

        // du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        float du_dx =
            SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x_squared);

        // d/dx [0.5 * x * (1 + tanh(u))]
        float dgelu_dx =
            0.5f * (1.0f + tanh_u)
            + 0.5f * value * (1.0f - tanh_u * tanh_u) * du_dx;

        dx[i] = dy[i] * dgelu_dx;
    }
}

# Step 12 - softmax_rows_forward_kernel
__global__ void softmax_rows_forward_kernel(const float* logits, float* probs, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    extern __shared__ float shared[];

    const float* row_logits = logits + row * N;
    float* row_probs = probs + row * N;

    // Find the maximum value in the row.
    // Initialize from the first element when possible, avoiding
    // dependencies on FLT_MAX or CUDART_INF_F.
    float local_max = 0.0f;

    if (N > 0) {
        local_max = row_logits[0];

        for (int j = tid; j < N; j += blockDim.x) {
            local_max = fmaxf(local_max, row_logits[j]);
        }
    }

    shared[tid] = local_max;
    __syncthreads();

    // Max reduction.
    int active = blockDim.x;

    while (active > 1) {
        int half = (active + 1) / 2;

        if (tid < half) {
            int other = tid + half;

            if (other < active) {
                shared[tid] = fmaxf(shared[tid], shared[other]);
            }
        }

        __syncthreads();
        active = half;
    }

    float row_max = shared[0];
    __syncthreads();

    // Compute sum of exp(logit - max).
    float local_sum = 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        local_sum += expf(row_logits[j] - row_max);
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Sum reduction.
    active = blockDim.x;

    while (active > 1) {
        int half = (active + 1) / 2;

        if (tid < half) {
            int other = tid + half;

            if (other < active) {
                shared[tid] += shared[other];
            }
        }

        __syncthreads();
        active = half;
    }

    float row_sum = shared[0];
    __syncthreads();

    // Normalize.
    for (int j = tid; j < N; j += blockDim.x) {
        row_probs[j] = expf(row_logits[j] - row_max) / row_sum;
    }
}

# Step 13 - softmax_rows_backward_kernel
__global__ void softmax_rows_backward_kernel(
    const float* probs,
    const float* dprobs,
    float* dlogits,
    int M,
    int N
) {
    // One block processes one row.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    extern __shared__ float shared[];

    const float* row_probs = probs + row * N;
    const float* row_dprobs = dprobs + row * N;
    float* row_dlogits = dlogits + row * N;

    // Compute the row-specific scalar:
    // s = sum_j probs[j] * dprobs[j]
    float local_sum = 0.0f;

    // Stride across the row so this works even when N > blockDim.x.
    for (int j = tid; j < N; j += blockDim.x) {
        local_sum += row_probs[j] * row_dprobs[j];
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Reduce the partial sums across the block.
    int active = blockDim.x;

    while (active > 1) {
        int half = (active + 1) / 2;

        if (tid < half) {
            int other = tid + half;

            if (other < active) {
                shared[tid] += shared[other];
            }
        }

        __syncthreads();
        active = half;
    }

    float s = shared[0];
    __syncthreads();

    // dlogits[j] = probs[j] * (dprobs[j] - s)
    for (int j = tid; j < N; j += blockDim.x) {
        row_dlogits[j] = row_probs[j] * (row_dprobs[j] - s);
    }
}

# Step 14 - topk_per_row_kernel
__global__ void topk_per_row_kernel(
    const float* values,
    float* topk_values,
    int* topk_indices,
    int M,
    int N,
    int K
) {
    // One thread processes one complete row.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) {
        return;
    }

    // K is guaranteed to be <= 4.
    constexpr int MAX_K = 4;

    float best_values[MAX_K];
    int best_indices[MAX_K];

    // Initialize the top-K slots.
    for (int k = 0; k < MAX_K; ++k) {
        best_values[k] = -1.0e30f;
        best_indices[k] = -1;
    }

    const float* row_values = values + row * N;

    // Scan the entire row serially.
    for (int j = 0; j < N; ++j) {
        float value = row_values[j];

        // Find the insertion position for this value.
        int pos = K;

        for (int k = 0; k < K; ++k) {
            if (value > best_values[k]) {
                pos = k;
                break;
            }
        }

        // Insert into the sorted top-K list.
        if (pos < K) {
            for (int k = K - 1; k > pos; --k) {
                best_values[k] = best_values[k - 1];
                best_indices[k] = best_indices[k - 1];
            }

            best_values[pos] = value;
            best_indices[pos] = j;
        }
    }

    // Write the selected values and indices in descending order.
    for (int k = 0; k < K; ++k) {
        topk_values[row * K + k] = best_values[k];
        topk_indices[row * K + k] = best_indices[k];
    }
}

# Step 15 - normalize_topk_gates_kernel
__global__ void normalize_topk_gates_kernel(
    const float* topk_values,
    float* gates,
    int M,
    int K
) {
    // One block processes one row.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    extern __shared__ float shared[];

    const float* row_values = topk_values + row * K;
    float* row_gates = gates + row * K;

    // Each thread contributes one or more K entries.
    float local_sum = 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        local_sum += row_values[k];
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Reduce the sum across the block.
    int active = blockDim.x;

    while (active > 1) {
        int half = active / 2;

        if (tid < half) {
            shared[tid] += shared[tid + half];
        }

        __syncthreads();
        active = half;
    }

    float sum = shared[0];
    __syncthreads();

    // Normalize each selected gate.
    for (int k = tid; k < K; k += blockDim.x) {
        row_gates[k] = row_values[k] / sum;
    }
}

# Step 16 - normalize_topk_gates_backward_kernel
__global__ void normalize_topk_gates_backward_kernel(
    const float* topk_values,
    const float* gates,
    const float* dgates,
    float* dtopk_values,
    int M,
    int K
) {
    // One block processes one token/row.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    // Two shared-memory buffers:
    // shared[0 .. blockDim.x-1]              -> reduction for sum(v)
    // shared[blockDim.x .. 2*blockDim.x-1]   -> reduction for sum(dgates * values)
    extern __shared__ float shared[];

    float* sum_shared = shared;
    float* dot_shared = shared + blockDim.x;

    const float* row_values = topk_values + row * K;
    const float* row_gates = gates + row * K;
    const float* row_dgates = dgates + row * K;
    float* row_dvalues = dtopk_values + row * K;

    // ------------------------------------------------------------
    // Forward normalization:
    //   g_k = v_k / S
    //   S = sum_j v_j
    //
    // Backward:
    //   dL/dv_k = (dg_k - sum_j(dg_j * g_j)) / S
    // because g_j = v_j / S.
    // ------------------------------------------------------------

    float local_sum = 0.0f;
    float local_dot = 0.0f;

    for (int k = tid; k < K; k += blockDim.x) {
        local_sum += row_values[k];
        local_dot += row_dgates[k] * row_gates[k];
    }

    sum_shared[tid] = local_sum;
    dot_shared[tid] = local_dot;
    __syncthreads();

    // ------------------------------------------------------------
    // Reduce sum(v).
    // ------------------------------------------------------------
    int active = blockDim.x;

    while (active > 1) {
        int half = active / 2;

        if (tid < half) {
            sum_shared[tid] += sum_shared[tid + half];
            dot_shared[tid] += dot_shared[tid + half];
        }

        __syncthreads();
        active = half;
    }

    float sum = sum_shared[0];
    float dot = dot_shared[0];

    __syncthreads();

    // If the raw values sum to zero, gradients are defined as zero
    // according to the problem statement.
    if (sum == 0.0f) {
        for (int k = tid; k < K; k += blockDim.x) {
            row_dvalues[k] = 0.0f;
        }
        return;
    }

    // Compute gradient with respect to the raw top-K values.
    for (int k = tid; k < K; k += blockDim.x) {
        row_dvalues[k] = (row_dgates[k] - dot) / sum;
    }
}

# Step 17 - router_logits_forward (not yet solved)
# TODO: implement

# Step 18 - router_softmax_forward (not yet solved)
# TODO: implement

# Step 19 - router_topk_experts (not yet solved)
# TODO: implement

# Step 20 - router_gate_weight_backward (not yet solved)
# TODO: implement

# Step 21 - count_tokens_per_expert_kernel (not yet solved)
# TODO: implement

# Step 22 - expert_offsets_prefix_sum_kernel (not yet solved)
# TODO: implement

# Step 23 - assign_token_slots_kernel (not yet solved)
# TODO: implement

# Step 24 - gather_tokens_to_experts_kernel (not yet solved)
# TODO: implement

# Step 25 - scatter_grads_to_tokens_kernel (not yet solved)
# TODO: implement

# Step 26 - combine_expert_outputs_kernel (not yet solved)
# TODO: implement

# Step 27 - combine_backward_to_expert_outputs_kernel (not yet solved)
# TODO: implement

# Step 28 - combine_backward_to_gates_kernel (not yet solved)
# TODO: implement

# Step 29 - expert_up_projection_forward (not yet solved)
# TODO: implement

# Step 30 - expert_up_projection_add_bias (not yet solved)
# TODO: implement

# Step 31 - expert_hidden_activation_forward (not yet solved)
# TODO: implement

# Step 32 - expert_down_projection_forward (not yet solved)
# TODO: implement

# Step 33 - expert_down_projection_add_bias (not yet solved)
# TODO: implement

# Step 34 - expert_down_projection_backward_input (not yet solved)
# TODO: implement

# Step 35 - expert_down_projection_backward_weight (not yet solved)
# TODO: implement

# Step 36 - expert_down_projection_backward_bias (not yet solved)
# TODO: implement

# Step 37 - expert_activation_backward (not yet solved)
# TODO: implement

# Step 38 - expert_up_projection_backward_input (not yet solved)
# TODO: implement

# Step 39 - expert_up_projection_backward_weight (not yet solved)
# TODO: implement

# Step 40 - expert_up_projection_backward_bias (not yet solved)
# TODO: implement

# Step 41 - compute_dispatch_fractions (not yet solved)
# TODO: implement

# Step 42 - compute_mean_router_probs (not yet solved)
# TODO: implement

# Step 43 - load_balancing_aux_loss_forward (not yet solved)
# TODO: implement

# Step 44 - load_balancing_aux_loss_backward (not yet solved)
# TODO: implement

# Step 45 - mse_loss_forward (not yet solved)
# TODO: implement

# Step 46 - mse_loss_backward (not yet solved)
# TODO: implement

# Step 47 - zero_buffer (not yet solved)
# TODO: implement

# Step 48 - sgd_update_parameters (not yet solved)
# TODO: implement

# Step 49 - moe_forward (not yet solved)
# TODO: implement

# Step 50 - moe_backward (not yet solved)
# TODO: implement

# Step 51 - moe_training_step (not yet solved)
# TODO: implement

# Step 52 - moe_training_loop (not yet solved)
# TODO: implement

