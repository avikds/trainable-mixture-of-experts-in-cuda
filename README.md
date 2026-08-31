# Trainable Mixture of Experts in CUDA

Build a trainable Mixture-of-Experts layer from scratch in CUDA, starting from low-level matmul and activation kernels and culminating in a full forward, backward, and training loop with load-balancing auxiliary loss. The project gives you hands-on experience with sparse routing, token dispatch, and end-to-end MoE training on the GPU.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** matmul_naive_kernel
- [x] **2.** matmul_tiled_kernel
- [x] **3.** matmul_at_b_kernel
- [x] **4.** matmul_a_bt_kernel
- [x] **5.** add_bias_row_kernel
- [x] **6.** reduce_rows_to_bias_grad_kernel
- [x] **7.** elementwise_add_kernel
- [x] **8.** relu_forward_kernel
- [x] **9.** relu_backward_kernel
- [x] **10.** gelu_forward_kernel
- [x] **11.** gelu_backward_kernel
- [x] **12.** softmax_rows_forward_kernel
- [x] **13.** softmax_rows_backward_kernel
- [x] **14.** topk_per_row_kernel
- [x] **15.** normalize_topk_gates_kernel
- [x] **16.** normalize_topk_gates_backward_kernel
- [x] **17.** router_logits_forward
- [x] **18.** router_softmax_forward
- [x] **19.** router_topk_experts
- [x] **20.** router_gate_weight_backward
- [x] **21.** count_tokens_per_expert_kernel
- [x] **22.** expert_offsets_prefix_sum_kernel
- [x] **23.** assign_token_slots_kernel
- [x] **24.** gather_tokens_to_experts_kernel
- [x] **25.** scatter_grads_to_tokens_kernel
- [x] **26.** combine_expert_outputs_kernel
- [x] **27.** combine_backward_to_expert_outputs_kernel
- [x] **28.** combine_backward_to_gates_kernel
- [x] **29.** expert_up_projection_forward
- [x] **30.** expert_up_projection_add_bias
- [x] **31.** expert_hidden_activation_forward
- [x] **32.** expert_down_projection_forward
- [x] **33.** expert_down_projection_add_bias
- [x] **34.** expert_down_projection_backward_input
- [x] **35.** expert_down_projection_backward_weight
- [x] **36.** expert_down_projection_backward_bias
- [x] **37.** expert_activation_backward
- [x] **38.** expert_up_projection_backward_input
- [x] **39.** expert_up_projection_backward_weight
- [x] **40.** expert_up_projection_backward_bias
- [x] **41.** compute_dispatch_fractions
- [x] **42.** compute_mean_router_probs
- [x] **43.** load_balancing_aux_loss_forward
- [x] **44.** load_balancing_aux_loss_backward
- [x] **45.** mse_loss_forward
- [x] **46.** mse_loss_backward
- [x] **47.** zero_buffer
- [x] **48.** sgd_update_parameters
- [x] **49.** moe_forward
- [x] **50.** moe_backward
- [x] **51.** moe_training_step
- [x] **52.** moe_training_loop

## Results

```
Initial output[0]: 0.0005 -0.0000 -0.0005 -0.0014 
Target[0]:        0.2253 -0.4079 0.2751 0.0486 

Loss history:
  step  0  loss=0.378676
  step  1  loss=0.377430
  step  2  loss=0.376198
  step  3  loss=0.374978
  step  4  loss=0.373772
  step  5  loss=0.372578
  step  6  loss=0.371396
  step  7  loss=0.370226
  step  8  loss=0.369069
  step  9  loss=0.367923

Final output[0]: -0.0203 0.0207 0.0145 0.0129
```
