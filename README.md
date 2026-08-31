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
- [ ] **4.** matmul_a_bt_kernel
- [ ] **5.** add_bias_row_kernel
- [ ] **6.** reduce_rows_to_bias_grad_kernel
- [ ] **7.** elementwise_add_kernel
- [ ] **8.** relu_forward_kernel
- [ ] **9.** relu_backward_kernel
- [ ] **10.** gelu_forward_kernel
- [ ] **11.** gelu_backward_kernel
- [ ] **12.** softmax_rows_forward_kernel
- [ ] **13.** softmax_rows_backward_kernel
- [ ] **14.** topk_per_row_kernel
- [ ] **15.** normalize_topk_gates_kernel
- [ ] **16.** normalize_topk_gates_backward_kernel
- [ ] **17.** router_logits_forward
- [ ] **18.** router_softmax_forward
- [ ] **19.** router_topk_experts
- [ ] **20.** router_gate_weight_backward
- [ ] **21.** count_tokens_per_expert_kernel
- [ ] **22.** expert_offsets_prefix_sum_kernel
- [ ] **23.** assign_token_slots_kernel
- [ ] **24.** gather_tokens_to_experts_kernel
- [ ] **25.** scatter_grads_to_tokens_kernel
- [ ] **26.** combine_expert_outputs_kernel
- [ ] **27.** combine_backward_to_expert_outputs_kernel
- [ ] **28.** combine_backward_to_gates_kernel
- [ ] **29.** expert_up_projection_forward
- [ ] **30.** expert_up_projection_add_bias
- [ ] **31.** expert_hidden_activation_forward
- [ ] **32.** expert_down_projection_forward
- [ ] **33.** expert_down_projection_add_bias
- [ ] **34.** expert_down_projection_backward_input
- [ ] **35.** expert_down_projection_backward_weight
- [ ] **36.** expert_down_projection_backward_bias
- [ ] **37.** expert_activation_backward
- [ ] **38.** expert_up_projection_backward_input
- [ ] **39.** expert_up_projection_backward_weight
- [ ] **40.** expert_up_projection_backward_bias
- [ ] **41.** compute_dispatch_fractions
- [ ] **42.** compute_mean_router_probs
- [ ] **43.** load_balancing_aux_loss_forward
- [ ] **44.** load_balancing_aux_loss_backward
- [ ] **45.** mse_loss_forward
- [ ] **46.** mse_loss_backward
- [ ] **47.** zero_buffer
- [ ] **48.** sgd_update_parameters
- [ ] **49.** moe_forward
- [ ] **50.** moe_backward
- [ ] **51.** moe_training_step
- [ ] **52.** moe_training_loop

---

Built on Deep-ML.
