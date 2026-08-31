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

# Step 17 - router_logits_forward
void router_logits_forward(
    const float* X,
    const float* Wg,
    float* logits,
    int T,
    int D,
    int E
) {
    // Use 16x16 blocks, matching matmul_tiled_kernel.
    dim3 block(16, 16);
    dim3 grid(
        (E + 15) / 16,
        (T + 15) / 16
    );

    // Compute:
    // X     : T x D
    // Wg    : D x E
    // logits: T x E
    matmul_tiled_kernel<<<grid, block>>>(
        X,
        Wg,
        logits,
        T,
        E,
        D
    );
}

# Step 18 - router_softmax_forward
void router_softmax_forward(
    const float* logits,
    float* probs,
    int T,
    int E
) {
    // One block processes one token/row.
    // 128 threads provide enough parallelism while allowing
    // blockDim.x-strided access when E > 128.
    int threads = 128;

    // The softmax kernel uses one float of dynamic shared memory
    // per thread for its row reductions.
    size_t shared_bytes = threads * sizeof(float);

    softmax_rows_forward_kernel<<<T, threads, shared_bytes>>>(
        logits,
        probs,
        T,
        E
    );
}

# Step 19 - router_topk_experts
void router_topk_experts(
    const float* probs,
    float* topk_probs,
    int* topk_experts,
    int T,
    int E,
    int K
) {
    // One thread processes one token/row.
    int threads = 64;
    int blocks = (T + threads - 1) / threads;

    topk_per_row_kernel<<<blocks, threads>>>(
        probs,
        topk_probs,
        topk_experts,
        T,
        E,
        K
    );
}

# Step 20 - router_gate_weight_backward
void router_gate_weight_backward(
    const float* X,
    const float* dlogits,
    float* dWg,
    int T,
    int D,
    int E
) {
    // X       : T x D
    // dlogits : T x E
    // dWg     : D x E
    //
    // matmul_at_b_kernel expects:
    // A: K x M
    // B: K x N
    // C: M x N
    //
    // Set:
    // K = T, M = D, N = E
    // so C = X^T * dlogits.

    dim3 block(16, 16);
    dim3 grid(
        (E + 15) / 16,
        (D + 15) / 16
    );

    matmul_at_b_kernel<<<grid, block>>>(
        X,
        dlogits,
        dWg,
        D,
        E,
        T
    );
}

# Step 21 - count_tokens_per_expert_kernel
__global__ void count_tokens_per_expert_kernel(
    const int* topk_experts,
    int* expert_counts,
    int T,
    int K,
    int E
) {
    // One thread handles one (token, top-k slot) assignment.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * K;

    if (idx >= total) {
        return;
    }

    int expert = topk_experts[idx];

    // Guard against invalid expert ids.
    if (expert >= 0 && expert < E) {
        atomicAdd(&expert_counts[expert], 1);
    }
}

# Step 22 - expert_offsets_prefix_sum_kernel
__global__ void expert_offsets_prefix_sum_kernel(
    const int* expert_counts,
    int* expert_offsets,
    int E
) {
    // A single thread computes the exclusive prefix sum serially.
    if (threadIdx.x != 0) {
        return;
    }

    expert_offsets[0] = 0;

    int running_sum = 0;

    for (int e = 0; e < E; ++e) {
        running_sum += expert_counts[e];
        expert_offsets[e + 1] = running_sum;
    }
}

# Step 23 - assign_token_slots_kernel
__global__ void assign_token_slots_kernel(
    const int* topk_experts,
    const int* expert_offsets,
    int* slot_token_idx,
    int* slot_k_idx,
    int* expert_fill,
    int T,
    int K,
    int E
) {
    // One thread handles one (token, k) assignment.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * K;

    if (idx >= total) {
        return;
    }

    // Recover token index and top-k slot.
    int token = idx / K;
    int k = idx % K;

    int expert = topk_experts[idx];

    // Guard against invalid expert ids.
    if (expert < 0 || expert >= E) {
        return;
    }

    // Atomically claim the next available slot in this expert's bucket.
    int local_offset = atomicAdd(&expert_fill[expert], 1);

    // Convert the expert-local offset into a global slot index.
    int slot = expert_offsets[expert] + local_offset;

    // Record the source token and its top-k position.
    slot_token_idx[slot] = token;
    slot_k_idx[slot] = k;
}

# Step 24 - gather_tokens_to_experts_kernel
__global__ void gather_tokens_to_experts_kernel(
    const float* X,
    const int* slot_token_idx,
    float* X_dispatched,
    int total_slots,
    int D
) {
    // One thread handles one (slot, feature) element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_slots * D;

    if (idx >= total) {
        return;
    }

    int slot = idx / D;
    int d = idx % D;

    // Find the source token for this dispatched slot.
    int token = slot_token_idx[slot];

    // Copy the corresponding feature.
    X_dispatched[slot * D + d] = X[token * D + d];
}

# Step 25 - scatter_grads_to_tokens_kernel
__global__ void scatter_grads_to_tokens_kernel(
    const float* dX_dispatched,
    const int* slot_token_idx,
    const float* gate_values,
    float* dX,
    int total_slots,
    int D
) {
    // One thread handles one (slot, feature) element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_slots * D;

    if (idx >= total) {
        return;
    }

    int slot = idx / D;
    int d = idx % D;

    // Find the original token associated with this slot.
    int token = slot_token_idx[slot];

    // Scale the expert gradient by the corresponding gate value.
    float gate = gate_values[slot];

    // Multiple slots can map to the same token, so atomicAdd is
    // required to safely accumulate their contributions.
    atomicAdd(
        &dX[token * D + d],
        gate * dX_dispatched[slot * D + d]
    );
}

# Step 26 - combine_expert_outputs_kernel
__global__ void combine_expert_outputs_kernel(
    const float* Y_dispatched,
    const int* slot_token_idx,
    const int* slot_k_idx,
    const float* gates,
    float* Y,
    int total_slots,
    int K,
    int D_out
) {
    // Each thread handles one (slot, output-feature) pair.
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard both dimensions.
    if (slot >= total_slots || d >= D_out) {
        return;
    }

    // Recover the destination token and its top-k gate index.
    int token = slot_token_idx[slot];
    int k = slot_k_idx[slot];

    // Gate value for this token/slot.
    float gate = gates[token * K + k];

    // Weighted expert output contribution.
    float contribution = gate * Y_dispatched[slot * D_out + d];

    // Multiple slots can contribute to the same token/output element,
    // so atomically accumulate the result.
    atomicAdd(
        &Y[token * D_out + d],
        contribution
    );
}

# Step 27 - combine_backward_to_expert_outputs_kernel
__global__ void combine_backward_to_expert_outputs_kernel(
    const float* dY,
    const int* slot_token_idx,
    const int* slot_k_idx,
    const float* gates,
    float* dY_dispatched,
    int total_slots,
    int K,
    int D_out
) {
    // One block processes one slot.
    int slot = blockIdx.x;
    int tid = threadIdx.x;

    if (slot >= total_slots) {
        return;
    }

    // Find the source token and its top-k gate index.
    int token = slot_token_idx[slot];
    int k = slot_k_idx[slot];

    // Retrieve the gate associated with this slot.
    float gate = gates[token * K + k];

    // Threads cooperate over the output features.
    for (int d = tid; d < D_out; d += blockDim.x) {
        dY_dispatched[slot * D_out + d] =
            gate * dY[token * D_out + d];
    }
}

# Step 28 - combine_backward_to_gates_kernel
__global__ void combine_backward_to_gates_kernel(
    const float* dY,
    const float* Y_dispatched,
    const int* slot_token_idx,
    const int* slot_k_idx,
    float* dgates,
    int total_slots,
    int K,
    int D_out
) {
    // One block processes one dispatched slot.
    int slot = blockIdx.x;
    int tid = threadIdx.x;

    if (slot >= total_slots) {
        return;
    }

    // Dynamic shared memory for the per-thread partial dot products.
    extern __shared__ float shared[];

    int token = slot_token_idx[slot];
    int k = slot_k_idx[slot];

    const float* dy_row = dY + token * D_out;
    const float* y_row = Y_dispatched + slot * D_out;

    // Each thread computes a partial dot product.
    float local_sum = 0.0f;

    for (int d = tid; d < D_out; d += blockDim.x) {
        local_sum += dy_row[d] * y_row[d];
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Reduce the partial dot products.
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

    // One value remains: dot(dY[token, :], Y_dispatched[slot, :]).
    if (tid == 0) {
        atomicAdd(
            &dgates[token * K + k],
            shared[0]
        );
    }
}

# Step 29 - expert_up_projection_forward
void expert_up_projection_forward(
    const float* X_e,
    const float* W_up,
    float* H_pre,
    int N_e,
    int D,
    int H
) {
    // No work is needed when the expert has no assigned tokens.
    if (N_e == 0) {
        return;
    }

    // X_e : N_e x D
    // W_up: D   x H
    // H_pre: N_e x H
    dim3 block(16, 16);
    dim3 grid(
        (H + 15) / 16,
        (N_e + 15) / 16
    );

    matmul_tiled_kernel<<<grid, block>>>(
        X_e,
        W_up,
        H_pre,
        N_e,
        H,
        D
    );
}

# Step 30 - expert_up_projection_add_bias
void expert_up_projection_add_bias(
    float* H_pre,
    const float* b_up,
    int N_e,
    int H
) {
    // No work is needed when the expert has no assigned tokens.
    if (N_e == 0) {
        return;
    }

    // H_pre: N_e x H
    // b_up : H
    // One thread handles one matrix element.
    dim3 block(16, 16);
    dim3 grid(
        (H + 15) / 16,
        (N_e + 15) / 16
    );

    add_bias_row_kernel<<<grid, block>>>(
        H_pre,
        b_up,
        N_e,
        H
    );
}

# Step 31 - expert_hidden_activation_forward
void expert_hidden_activation_forward(
    const float* d_hidden_pre,
    float* d_hidden_post,
    int num_tokens,
    int hidden_dim
) {
    int n = num_tokens * hidden_dim;

    // Nothing to do for an empty expert bucket.
    if (n == 0) {
        return;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    gelu_forward_kernel<<<blocks, threads>>>(
        d_hidden_pre,
        d_hidden_post,
        n
    );
}

# Step 32 - expert_down_projection_forward
void expert_down_projection_forward(
    const float* d_hidden_post,
    const float* d_w_down,
    float* d_output,
    int num_tokens,
    int hidden_dim,
    int out_dim
) {
    // Nothing to do for an empty expert bucket.
    if (num_tokens == 0) {
        return;
    }

    // d_hidden_post: num_tokens x hidden_dim
    // d_w_down     : hidden_dim x out_dim
    // d_output     : num_tokens x out_dim

    dim3 block(16, 16);
    dim3 grid(
        (out_dim + 15) / 16,
        (num_tokens + 15) / 16
    );

    matmul_tiled_kernel<<<grid, block>>>(
        d_hidden_post,
        d_w_down,
        d_output,
        num_tokens,
        out_dim,
        hidden_dim
    );
}

# Step 33 - expert_down_projection_add_bias
void expert_down_projection_add_bias(
    float* d_output,
    const float* d_b_down,
    int num_tokens,
    int out_dim
) {
    // Nothing to do for an empty expert bucket.
    if (num_tokens == 0) {
        return;
    }

    // d_output: num_tokens x out_dim
    // d_b_down: out_dim
    // One thread handles one matrix element.
    dim3 block(16, 16);
    dim3 grid(
        (out_dim + 15) / 16,
        (num_tokens + 15) / 16
    );

    add_bias_row_kernel<<<grid, block>>>(
        d_output,
        d_b_down,
        num_tokens,
        out_dim
    );
}

# Step 34 - expert_down_projection_backward_input
void expert_down_projection_backward_input(
    const float* d_grad_output,
    const float* d_w_down,
    float* d_grad_hidden_post,
    int num_tokens,
    int hidden_dim,
    int out_dim
) {
    // Nothing to do for an empty expert bucket.
    if (num_tokens == 0) {
        return;
    }

    // d_grad_output      : num_tokens x out_dim
    // d_w_down           : hidden_dim x out_dim
    // d_grad_hidden_post : num_tokens x hidden_dim
    //
    // Compute:
    // d_grad_hidden_post = d_grad_output @ d_w_down^T
    //
    // matmul_a_bt_kernel computes:
    // C = A @ B^T
    // where A is M x K and B is N x K.
    //
    // Set:
    // M = num_tokens
    // N = hidden_dim
    // K = out_dim

    dim3 block(16, 16);
    dim3 grid(
        (hidden_dim + 15) / 16,
        (num_tokens + 15) / 16
    );

    matmul_a_bt_kernel<<<grid, block>>>(
        d_grad_output,
        d_w_down,
        d_grad_hidden_post,
        num_tokens,
        hidden_dim,
        out_dim
    );
}

# Step 35 - expert_down_projection_backward_weight
void expert_down_projection_backward_weight(
    const float* d_hidden_post,
    const float* d_grad_output,
    float* d_grad_w_down,
    int num_tokens,
    int hidden_dim,
    int out_dim
) {
    // Nothing to do for an empty expert bucket.
    if (num_tokens == 0) {
        return;
    }

    // d_hidden_post : num_tokens x hidden_dim
    // d_grad_output : num_tokens x out_dim
    // d_grad_w_down : hidden_dim x out_dim
    //
    // Compute:
    // d_grad_w_down = d_hidden_post^T @ d_grad_output
    //
    // matmul_at_b_kernel computes:
    // C = A^T @ B
    // where A is K x M and B is K x N.
    //
    // Set:
    // K = num_tokens
    // M = hidden_dim
    // N = out_dim

    dim3 block(16, 16);
    dim3 grid(
        (out_dim + 15) / 16,
        (hidden_dim + 15) / 16
    );

    matmul_at_b_kernel<<<grid, block>>>(
        d_hidden_post,
        d_grad_output,
        d_grad_w_down,
        hidden_dim,
        out_dim,
        num_tokens
    );
}

# Step 36 - expert_down_projection_backward_bias
void expert_down_projection_backward_bias(
    const float* d_grad_output,
    float* d_grad_b_down,
    int num_tokens,
    int out_dim
) {
    // For an empty expert bucket, the bias gradient is zero.
    if (num_tokens == 0) {
        cudaMemset(
            d_grad_b_down,
            0,
            out_dim * sizeof(float)
        );
        return;
    }

    // One thread handles one output dimension/column.
    int threads = 128;
    int blocks = (out_dim + threads - 1) / threads;

    reduce_rows_to_bias_grad_kernel<<<blocks, threads>>>(
        d_grad_output,
        d_grad_b_down,
        num_tokens,
        out_dim
    );
}

# Step 37 - expert_activation_backward
void expert_activation_backward(
    const float* d_hidden_pre,
    const float* d_grad_hidden_post,
    float* d_grad_hidden_pre,
    int num_tokens,
    int hidden_dim
) {
    int n = num_tokens * hidden_dim;

    // Nothing to do for an empty expert bucket.
    if (n == 0) {
        return;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // The expert hidden activation is GELU, so use the GELU backward kernel.
    gelu_backward_kernel<<<blocks, threads>>>(
        d_hidden_pre,
        d_grad_hidden_post,
        d_grad_hidden_pre,
        n
    );
}

# Step 38 - expert_up_projection_backward_input
void expert_up_projection_backward_input(
    const float* d_grad_hidden_pre,
    const float* d_w_up,
    float* d_grad_input,
    int num_tokens,
    int in_dim,
    int hidden_dim
) {
    // Nothing to do for an empty expert bucket.
    if (num_tokens == 0) {
        return;
    }

    // d_grad_hidden_pre: num_tokens x hidden_dim
    // d_w_up:            in_dim x hidden_dim
    // d_grad_input:      num_tokens x in_dim
    //
    // Compute:
    // d_grad_input = d_grad_hidden_pre @ d_w_up^T
    //
    // matmul_a_bt_kernel computes:
    // C = A @ B^T
    // where A is M x K and B is N x K.
    //
    // Set:
    // M = num_tokens
    // N = in_dim
    // K = hidden_dim

    dim3 block(16, 16);
    dim3 grid(
        (in_dim + 15) / 16,
        (num_tokens + 15) / 16
    );

    matmul_a_bt_kernel<<<grid, block>>>(
        d_grad_hidden_pre,
        d_w_up,
        d_grad_input,
        num_tokens,
        in_dim,
        hidden_dim
    );
}

# Step 39 - expert_up_projection_backward_weight
void expert_up_projection_backward_weight(
    const float* d_input,
    const float* d_grad_hidden_pre,
    float* d_grad_w_up,
    int num_tokens,
    int in_dim,
    int hidden_dim
) {
    // Nothing to do for an empty expert bucket.
    if (num_tokens == 0) {
        return;
    }

    // d_input            : num_tokens x in_dim
    // d_grad_hidden_pre  : num_tokens x hidden_dim
    // d_grad_w_up        : in_dim x hidden_dim
    //
    // Compute:
    // d_grad_w_up = d_input^T @ d_grad_hidden_pre
    //
    // matmul_at_b_kernel computes:
    // C = A^T @ B
    // where A is K x M and B is K x N.
    //
    // Set:
    // K = num_tokens
    // M = in_dim
    // N = hidden_dim

    dim3 block(16, 16);
    dim3 grid(
        (hidden_dim + 15) / 16,
        (in_dim + 15) / 16
    );

    matmul_at_b_kernel<<<grid, block>>>(
        d_input,
        d_grad_hidden_pre,
        d_grad_w_up,
        in_dim,
        hidden_dim,
        num_tokens
    );
}

# Step 40 - expert_up_projection_backward_bias
void expert_up_projection_backward_bias(
    const float* d_grad_hidden_pre,
    float* d_grad_b_up,
    int num_tokens,
    int hidden_dim
) {
    // For an empty expert bucket, the bias gradient is zero.
    if (num_tokens == 0) {
        cudaMemset(
            d_grad_b_up,
            0,
            hidden_dim * sizeof(float)
        );
        return;
    }

    // One thread handles one hidden dimension/column.
    int threads = 128;
    int blocks = (hidden_dim + threads - 1) / threads;

    // Sum the gradient across all tokens for each hidden dimension.
    reduce_rows_to_bias_grad_kernel<<<blocks, threads>>>(
        d_grad_hidden_pre,
        d_grad_b_up,
        num_tokens,
        hidden_dim
    );
}

# Step 41 - compute_dispatch_fractions
__global__ void dispatch_fractions_kernel(
    const int* counts,
    float* fractions,
    float denominator,
    int num_experts
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= num_experts) {
        return;
    }

    if (denominator > 0.0f) {
        fractions[e] = static_cast<float>(counts[e]) / denominator;
    } else {
        fractions[e] = 0.0f;
    }
}

void compute_dispatch_fractions(
    const int* d_expert_token_counts,
    float* d_dispatch_fractions,
    int num_tokens,
    int top_k,
    int num_experts
) {
    int total_assignments = num_tokens * top_k;

    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;

    dispatch_fractions_kernel<<<blocks, threads>>>(
        d_expert_token_counts,
        d_dispatch_fractions,
        static_cast<float>(total_assignments),
        num_experts
    );
}

# Step 42 - compute_mean_router_probs
__global__ void mean_router_probs_kernel(
    const float* router_probs,
    float* mean_probs,
    int num_tokens,
    int num_experts
) {
    // One thread handles one expert.
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= num_experts) {
        return;
    }

    // Handle the empty-token case safely.
    if (num_tokens == 0) {
        mean_probs[e] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // Sum this expert's probability across all tokens.
    for (int t = 0; t < num_tokens; ++t) {
        sum += router_probs[t * num_experts + e];
    }

    mean_probs[e] = sum / static_cast<float>(num_tokens);
}

void compute_mean_router_probs(
    const float* d_router_probs,
    float* d_mean_probs,
    int num_tokens,
    int num_experts
) {
    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;

    mean_router_probs_kernel<<<blocks, threads>>>(
        d_router_probs,
        d_mean_probs,
        num_tokens,
        num_experts
    );
}

# Step 43 - load_balancing_aux_loss_forward
__global__ void load_balancing_aux_loss_forward_kernel(
    const float* d_dispatch_fractions,
    const float* d_mean_probs,
    float* d_aux_loss,
    int num_experts
) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Each thread accumulates a partial sum over experts.
    float local_sum = 0.0f;

    for (int e = tid; e < num_experts; e += blockDim.x) {
        local_sum += d_dispatch_fractions[e] * d_mean_probs[e];
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Reduce partial sums.
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

    // Standard MoE load-balancing loss:
    // aux_loss = num_experts * sum_e(f_e * P_e)
    if (tid == 0) {
        d_aux_loss[0] =
            static_cast<float>(num_experts) * shared[0];
    }
}

void load_balancing_aux_loss_forward(
    const float* d_dispatch_fractions,
    const float* d_mean_probs,
    float* d_aux_loss,
    int num_experts
) {
    // Define the output for the degenerate case.
    if (num_experts <= 0) {
        cudaMemset(d_aux_loss, 0, sizeof(float));
        return;
    }

    int threads = 128;
    size_t shared_bytes = threads * sizeof(float);

    load_balancing_aux_loss_forward_kernel<<<1, threads, shared_bytes>>>(
        d_dispatch_fractions,
        d_mean_probs,
        d_aux_loss,
        num_experts
    );
}

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

