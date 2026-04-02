# Codebase Module Report

- Root: `/mnt/code/yehangcheng/verl`
- Source files scanned: `642`
- Modules identified: `642`

## Structure Tree

```text
├── docs
│   ├── _static
│   │   └── js
│   │       ├── resizable-sidebar.js
│   │       └── runllm-widget.js
│   └── conf.py
├── examples
│   ├── data_preprocess
│   │   ├── aime2024_multiturn_w_tool.py
│   │   ├── dapo_multiturn_w_tool.py
│   │   ├── full_hh_rlhf.py
│   │   ├── geo3k.py
│   │   ├── geo3k_multiturn_w_tool.py
│   │   ├── gsm8k.py
│   │   ├── gsm8k_multiturn_sft.py
│   │   ├── gsm8k_multiturn_w_interaction.py
│   │   ├── gsm8k_multiturn_w_tool.py
│   │   ├── gsm8k_tool_agent_loop.py
│   │   ├── hellaswag.py
│   │   ├── math_dataset.py
│   │   ├── multiturn.py
│   │   ├── pokemon.py
│   │   └── preprocess_search_r1_dataset.py
│   ├── fapo_trainer
│   │   ├── prepare_data.py
│   │   └── reward_fn.py
│   ├── flowgrpo_trainer
│   │   └── reward_fn.py
│   ├── sglang_multiturn
│   │   ├── gsm8k_toolcall_shaping
│   │   │   └── gsm8k_toolcall_shaping.py
│   │   └── search_r1_like
│   │       └── local_dense_retriever
│   │           ├── download.py
│   │           └── retrieval_server.py
│   ├── split_placement
│   │   ├── main_ppo_split.py
│   │   └── split_monkey_patch.py
│   ├── tutorial
│   │   └── agent_loop_get_started
│   │       └── sandbox.py
│   └── vllm_omni
│       ├── pipeline_qwenimage.py
│       └── scheduling_flow_match_sde_discrete.py
├── my_recipes
│   ├── genrm_remote
│   │   ├── reward_function.py
│   │   └── reward_function_genrm.py
│   ├── gym
│   │   └── prepare_qwen_gym_parquet.py
│   ├── insturct_following
│   │   ├── __init__.py
│   │   ├── evaluation_main.py
│   │   ├── instructions.py
│   │   ├── instructions_registry.py
│   │   ├── instructions_util.py
│   │   └── reward_function.py
│   ├── logic_rl
│   │   └── reward_fn.py
│   ├── rllm
│   │   ├── rewards
│   │   │   ├── code_utils
│   │   │   │   ├── codeforces.py
│   │   │   │   ├── firejail_exec.py
│   │   │   │   ├── humanevalplus.py
│   │   │   │   ├── kodcode.py
│   │   │   │   ├── livecodebench.py
│   │   │   │   ├── pyext2.py
│   │   │   │   ├── taco.py
│   │   │   │   └── utils.py
│   │   │   ├── math_utils
│   │   │   │   ├── __init__.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── code_reward.py
│   │   │   ├── math_reward.py
│   │   │   ├── reward_types.py
│   │   │   └── rl_reward.py
│   │   ├── tools
│   │   │   ├── code_tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── code_tool.py
│   │   │   │   ├── e2b_tool.py
│   │   │   │   ├── lcb_tool.py
│   │   │   │   ├── local_tool.py
│   │   │   │   ├── together_tool.py
│   │   │   │   └── utils.py
│   │   │   ├── math_tools
│   │   │   │   ├── __init__.py
│   │   │   │   └── calculator.py
│   │   │   ├── web_tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── firecrawl_tool.py
│   │   │   │   ├── gsearch_tool.py
│   │   │   │   └── tavily_tool.py
│   │   │   ├── __init__.py
│   │   │   ├── example_tool.py
│   │   │   ├── multi_tool.py
│   │   │   ├── tool_base.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── globals.py
│   │   ├── system_prompts.py
│   │   └── utils.py
│   ├── syntool
│   │   ├── __init__.py
│   │   ├── agent_loop.py
│   │   ├── dataset.py
│   │   ├── reward.py
│   │   └── tool.py
│   └── __init__.py
├── scripts
│   ├── veomni
│   │   ├── moe_merge.py
│   │   └── moe_split.py
│   ├── __init__.py
│   ├── converter_hf_to_mcore.py
│   ├── diagnose.py
│   ├── init_random_model.py
│   ├── legacy_model_merger.py
│   ├── megatron_merge_lora.py
│   ├── print_cfg.py
│   └── rollout_viewer.py
├── tests
│   ├── checkpoint_engine
│   │   ├── __init__.py
│   │   ├── test_correctness_on_gpu.py
│   │   ├── test_correctness_on_npu.py
│   │   ├── test_special_server_adapter.py
│   │   └── test_utils.py
│   ├── experimental
│   │   ├── agent_loop
│   │   │   ├── agent_utils.py
│   │   │   ├── test_agent_loop_extra_fields_schema_on_cpu.py
│   │   │   ├── test_basic_agent_loop.py
│   │   │   ├── test_gpt_oss_tool_parser.py
│   │   │   ├── test_multi_modal.py
│   │   │   └── test_standalone_rollout.py
│   │   ├── reward_loop
│   │   │   ├── reward_fn.py
│   │   │   ├── test_agent_reward_loop_colocate.py
│   │   │   ├── test_agent_reward_loop_standalone.py
│   │   │   ├── test_async_token_bucket_on_cpu.py
│   │   │   ├── test_math_verify.py
│   │   │   ├── test_rate_limited_reward_manager_on_cpu.py
│   │   │   ├── test_reward_model_disrm.py
│   │   │   ├── test_reward_model_genrm.py
│   │   │   └── test_visual_reward_manager.py
│   │   └── vla
│   │       └── test_sim_envs.py
│   ├── interactions
│   │   ├── __init__.py
│   │   ├── test_gsm8k_interaction.py
│   │   └── test_interaction_registry.py
│   ├── models
│   │   ├── test_engine.py
│   │   ├── test_liger_vl_compat.py
│   │   ├── test_tiled_mlp_accuracy.py
│   │   ├── test_transformer.py
│   │   └── test_transformers_ulysses.py
│   ├── my_recipes
│   │   └── test_syntool_recipe.py
│   ├── single_controller
│   │   ├── base
│   │   │   └── test_decorator.py
│   │   ├── check_worker_alive
│   │   │   └── main.py
│   │   ├── detached_worker
│   │   │   ├── client.py
│   │   │   └── server.py
│   │   ├── __init__.py
│   │   ├── test_auto_padding_on_cpu.py
│   │   ├── test_colocated_workers.py
│   │   ├── test_colocated_workers_fused.py
│   │   ├── test_data_transfer.py
│   │   ├── test_decorator_on_cpu.py
│   │   ├── test_device_mesh_register.py
│   │   ├── test_driverfunc_to_worker.py
│   │   ├── test_fused_workers_on_cpu.py
│   │   ├── test_get_set_dispatch_collect_cpu.py
│   │   ├── test_high_level_scheduling_api.py
│   │   ├── test_nested_worker.py
│   │   ├── test_ray_collectives.py
│   │   ├── test_ray_local_envs_on_cpu.py
│   │   ├── test_ray_utils_on_cpu.py
│   │   ├── test_rvdz.py
│   │   ├── test_split_resource_pool.py
│   │   ├── test_worker_group_basics.py
│   │   └── test_worker_group_torch.py
│   ├── special_distributed
│   │   ├── test_fsdp_ckpt.py
│   │   ├── test_mcore_config_converter.py
│   │   ├── test_tensor_dict.py
│   │   └── test_torch_functional.py
│   ├── special_e2e
│   │   ├── envs
│   │   │   ├── digit_completion
│   │   │   │   ├── __init__.py
│   │   │   │   ├── task.py
│   │   │   │   └── tokenizer.py
│   │   │   └── __init__.py
│   │   ├── sft
│   │   │   └── compare_sft_engine_results.py
│   │   ├── __init__.py
│   │   ├── check_custom_rwd_fn.py
│   │   └── check_results.py
│   ├── special_sanity
│   │   ├── check_api_docs.py
│   │   ├── check_dataproto_usage.py
│   │   ├── check_device_api_usage.py
│   │   ├── check_docs_time_info.py
│   │   ├── check_docstrings.py
│   │   ├── check_license.py
│   │   ├── check_pr_description.py
│   │   ├── check_pr_title.py
│   │   ├── test_config_docs.py
│   │   ├── test_import.py
│   │   ├── type_coverage_check.py
│   │   ├── validate_imported_docs.py
│   │   └── validate_structure.py
│   ├── special_standalone
│   │   └── test_memory_buffers.py
│   ├── trainer
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── test_algo_config_on_cpu.py
│   │   │   └── test_legacy_config_on_cpu.py
│   │   ├── ppo
│   │   │   ├── __init__.py
│   │   │   ├── test_core_algos_on_cpu.py
│   │   │   ├── test_metric_utils_on_cpu.py
│   │   │   ├── test_rollout_corr.py
│   │   │   └── test_rollout_corr_integration.py
│   │   └── __init__.py
│   ├── utils
│   │   ├── ckpt
│   │   │   ├── test_checkpoint_cleanup_on_cpu.py
│   │   │   └── test_esi_save_ckpt_on_cpu.py
│   │   ├── dataset
│   │   │   ├── test_create_rl_sampler_on_cpu.py
│   │   │   ├── test_multiturn_sft_dataset_on_cpu.py
│   │   │   ├── test_rl_collate_fn_on_cpu.py
│   │   │   └── test_rl_dataset_on_cpu.py
│   │   ├── debug
│   │   │   └── test_metrics.py
│   │   ├── megatron
│   │   │   └── test_pipeline_parallel.py
│   │   ├── reward_score
│   │   │   ├── reward_score
│   │   │   │   └── test_sandbox_fusion_on_cpu.py
│   │   │   └── test_sandbox_on_cpu.py
│   │   ├── _test_module.py
│   │   ├── test_activation_offload.py
│   │   ├── test_bucketed_weight_transfer.py
│   │   ├── test_check_ipc_version_support_on_npu.py
│   │   ├── test_check_profiler_output.py
│   │   ├── test_config_on_cpu.py
│   │   ├── test_flops_counter.py
│   │   ├── test_fs_on_cpu.py
│   │   ├── test_fsdp2_peft_wrapping.py
│   │   ├── test_fsdp_lora_merge.py
│   │   ├── test_groupwise.py
│   │   ├── test_import_utils_on_cpu.py
│   │   ├── test_linear_cross_entropy.py
│   │   ├── test_mlflow_key_sanitization.py
│   │   ├── test_model_on_cpu.py
│   │   ├── test_normalize_peft_param_name.py
│   │   ├── test_normalize_peft_param_name_on_cpu.py
│   │   ├── test_nvtx_profile.py
│   │   ├── test_padding_on_cpu.py
│   │   ├── test_prepare_micro_batches_with_group_size.py
│   │   ├── test_rollout_skip_on_cpu.py
│   │   ├── test_rollout_trace_on_cpu.py
│   │   ├── test_seqlen_balancing.py
│   │   ├── test_server_profiler.py
│   │   ├── test_shared_memory.py
│   │   ├── test_special_linear_cross_entropy_tp.py
│   │   ├── test_special_megatron_kl_loss_tp.py
│   │   ├── test_special_mstx_profile.py
│   │   ├── test_temp_env_on_cpu.py
│   │   ├── test_timeout_decorator_cpu.py
│   │   ├── test_tokenizer_normalize_on_cpu.py
│   │   ├── test_torch_functional.py
│   │   └── test_torch_profile.py
│   ├── workers
│   │   ├── actor
│   │   │   └── test_special_dp_actor.py
│   │   ├── config
│   │   │   ├── test_actor_config_on_cpu.py
│   │   │   ├── test_critic_config_on_cpu.py
│   │   │   ├── test_engine_config_on_cpu.py
│   │   │   ├── test_model_config_on_cpu.py
│   │   │   └── test_optim_config_on_cpu.py
│   │   ├── critic
│   │   │   └── test_special_dp_critic.py
│   │   ├── reward_manager
│   │   │   └── test_registry_on_cpu.py
│   │   ├── rollout
│   │   │   ├── perf
│   │   │   │   └── vllm_async_rollout.py
│   │   │   ├── rollout_sglang
│   │   │   │   ├── test_http_server_engine.py
│   │   │   │   └── test_lora_sleep_level.py
│   │   │   ├── rollout_trtllm
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_adapter.py
│   │   │   │   ├── test_async_server.py
│   │   │   │   └── test_trtllm_rollout_utils.py
│   │   │   ├── rollout_vllm
│   │   │   │   ├── run_fsdp_vllm.py
│   │   │   │   ├── test_vllm_abort.py
│   │   │   │   └── test_vllm_omni_generate.py
│   │   │   ├── test_hf_rollout.py
│   │   │   ├── test_sglang_async_rollout_multimodal_delta.py
│   │   │   ├── test_sglang_rollout_sharding_manager.py
│   │   │   └── test_vllm_cli_args_on_cpu.py
│   │   ├── test_fsdp_attn_implementation.py
│   │   └── test_fsdp_workers.py
│   ├── __init__.py
│   ├── test_base_config_on_cpu.py
│   ├── test_protocol_on_cpu.py
│   └── test_protocol_v2_on_cpu.py
├── verl
│   ├── checkpoint_engine
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hccl_checkpoint_engine.py
│   │   ├── kimi_checkpoint_engine.py
│   │   ├── mooncake_checkpoint_engine.py
│   │   ├── nccl_checkpoint_engine.py
│   │   └── nixl_checkpoint_engine.py
│   ├── experimental
│   │   ├── agent_loop
│   │   │   ├── __init__.py
│   │   │   ├── agent_loop.py
│   │   │   ├── prometheus_utils.py
│   │   │   ├── single_turn_agent_loop.py
│   │   │   ├── tool_agent_loop.py
│   │   │   ├── tool_parser.py
│   │   │   └── utils.py
│   │   ├── dataset
│   │   │   ├── __init__.py
│   │   │   └── sampler.py
│   │   ├── dynamic_dataset
│   │   │   ├── __init__.py
│   │   │   └── dynamicgen_dataset.py
│   │   ├── fully_async_policy
│   │   │   ├── agent_loop
│   │   │   │   ├── __init__.py
│   │   │   │   └── agent_loop.py
│   │   │   ├── unittest
│   │   │   │   └── simple_streaming_demo.py
│   │   │   ├── detach_utils.py
│   │   │   ├── fully_async_main.py
│   │   │   ├── fully_async_rollouter.py
│   │   │   ├── fully_async_trainer.py
│   │   │   └── message_queue.py
│   │   ├── one_step_off_policy
│   │   │   ├── main_ppo.py
│   │   │   └── ray_trainer.py
│   │   ├── reward_loop
│   │   │   ├── reward_manager
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── dapo.py
│   │   │   │   ├── gdpo.py
│   │   │   │   ├── limited.py
│   │   │   │   ├── naive.py
│   │   │   │   ├── registry.py
│   │   │   │   ├── remote.py
│   │   │   │   └── visual.py
│   │   │   ├── router
│   │   │   │   ├── inner_sglang_router.py
│   │   │   │   └── naive_router.py
│   │   │   ├── __init__.py
│   │   │   ├── reward_loop.py
│   │   │   └── reward_model.py
│   │   ├── separation
│   │   │   ├── __init__.py
│   │   │   ├── engine_workers.py
│   │   │   ├── ray_trainer.py
│   │   │   └── utils.py
│   │   ├── teacher_loop
│   │   │   ├── __init__.py
│   │   │   ├── teacher_manager.py
│   │   │   └── teacher_model.py
│   │   ├── vla
│   │   │   ├── envs
│   │   │   │   ├── isaac_env
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── isaac_env.py
│   │   │   │   ├── libero_env
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── libero_env.py
│   │   │   │   │   ├── utils.py
│   │   │   │   │   └── venv.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── action_utils.py
│   │   │   ├── models
│   │   │   │   ├── modules
│   │   │   │   │   └── mlp.py
│   │   │   │   ├── openvla_oft
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── configuration_prismatic.py
│   │   │   │   │   ├── constants.py
│   │   │   │   │   ├── modeling_prismatic.py
│   │   │   │   │   ├── processing_prismatic.py
│   │   │   │   │   └── train_utils.py
│   │   │   │   ├── pi0_torch
│   │   │   │   │   ├── model
│   │   │   │   │   │   ├── modeling_pi0.py
│   │   │   │   │   │   └── paligemma_with_expert.py
│   │   │   │   │   ├── policy
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── libero_policy.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── configuration_pi0_torch.py
│   │   │   │   │   ├── modeling_pi0_torch.py
│   │   │   │   │   └── pi0_utils.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── register_vla_models.py
│   │   │   ├── sac
│   │   │   │   ├── base.py
│   │   │   │   ├── naive_rollout_pi05.py
│   │   │   │   ├── replay_pool.py
│   │   │   │   ├── sac_actor.py
│   │   │   │   └── sac_ray_trainer.py
│   │   │   ├── workers
│   │   │   │   └── env
│   │   │   │       ├── env_loop_wg_test.py
│   │   │   │       ├── env_manager.py
│   │   │   │       └── env_worker.py
│   │   │   ├── dp_rob.py
│   │   │   ├── env_loop.py
│   │   │   ├── fsdp_workers.py
│   │   │   ├── main_ppo.py
│   │   │   ├── main_sac.py
│   │   │   ├── naive_rollout_rob.py
│   │   │   ├── prepare_libero_dataset.py
│   │   │   └── rob_ray_trainer.py
│   │   └── __init__.py
│   ├── interactions
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   └── interaction_registry.py
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gsm8k_interaction.py
│   │   ├── gym_env.py
│   │   ├── gym_interaction.py
│   │   └── weather_interaction.py
│   ├── model_merger
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── base_model_merger.py
│   │   ├── fsdp_model_merger.py
│   │   └── megatron_model_merger.py
│   ├── models
│   │   ├── mcore
│   │   │   ├── qwen2_5_vl
│   │   │   │   ├── __init__.py
│   │   │   │   ├── attention.py
│   │   │   │   ├── model.py
│   │   │   │   ├── rope_utils.py
│   │   │   │   ├── vision_config.py
│   │   │   │   ├── vision_model.py
│   │   │   │   └── vision_transformer_block.py
│   │   │   ├── __init__.py
│   │   │   ├── bridge.py
│   │   │   ├── config_converter.py
│   │   │   ├── loader.py
│   │   │   ├── mbridge.py
│   │   │   ├── model_forward.py
│   │   │   ├── model_forward_1f1b_overlap.py
│   │   │   ├── model_forward_fused.py
│   │   │   ├── model_initializer.py
│   │   │   ├── mtp_patch.py
│   │   │   ├── patch.py
│   │   │   ├── registry.py
│   │   │   ├── saver.py
│   │   │   ├── util.py
│   │   │   └── weight_converter.py
│   │   ├── transformers
│   │   │   ├── __init__.py
│   │   │   ├── apertus.py
│   │   │   ├── dense_common.py
│   │   │   ├── glm4v.py
│   │   │   ├── kimi_vl.py
│   │   │   ├── llama.py
│   │   │   ├── monkey_patch.py
│   │   │   ├── npu_patch.py
│   │   │   ├── qwen2.py
│   │   │   ├── qwen2_vl.py
│   │   │   ├── qwen3_vl.py
│   │   │   └── tiled_mlp.py
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── weight_loader_registry.py
│   ├── single_controller
│   │   ├── base
│   │   │   ├── __init__.py
│   │   │   ├── decorator.py
│   │   │   ├── worker.py
│   │   │   └── worker_group.py
│   │   ├── ray
│   │   │   ├── __init__.py
│   │   │   └── base.py
│   │   └── __init__.py
│   ├── third_party
│   │   ├── torch
│   │   │   ├── distributed
│   │   │   │   ├── checkpoint
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── state_dict.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── _state_dict_utils.py
│   │   │   └── __init__.py
│   │   ├── vllm
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── tools
│   │   ├── utils
│   │   │   ├── mcp_clients
│   │   │   │   ├── McpClientManager.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── search_r1_like_utils.py
│   │   │   └── tool_registry.py
│   │   ├── __init__.py
│   │   ├── base_tool.py
│   │   ├── geo3k_tool.py
│   │   ├── gsm8k_tool.py
│   │   ├── image_zoom_in_tool.py
│   │   ├── mcp_base_tool.py
│   │   ├── mcp_search_tool.py
│   │   ├── sandbox_fusion_tools.py
│   │   ├── schemas.py
│   │   └── search_tool.py
│   ├── trainer
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── algorithm.py
│   │   │   └── config.py
│   │   ├── distillation
│   │   │   ├── fsdp
│   │   │   │   └── losses.py
│   │   │   ├── megatron
│   │   │   │   └── losses.py
│   │   │   ├── __init__.py
│   │   │   └── losses.py
│   │   ├── ppo
│   │   │   ├── __init__.py
│   │   │   ├── core_algos.py
│   │   │   ├── metric_utils.py
│   │   │   ├── prefix_grouper_utils.py
│   │   │   ├── ray_trainer.py
│   │   │   ├── reward.py
│   │   │   ├── rollout_corr_helper.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── constants_ppo.py
│   │   ├── main_eval.py
│   │   ├── main_generation_server.py
│   │   ├── main_ppo.py
│   │   ├── sft_trainer.py
│   │   └── sft_trainer_ray.py
│   ├── utils
│   │   ├── checkpoint
│   │   │   ├── __init__.py
│   │   │   ├── checkpoint_handler.py
│   │   │   ├── checkpoint_manager.py
│   │   │   ├── fsdp_checkpoint_manager.py
│   │   │   └── megatron_checkpoint_manager.py
│   │   ├── dataset
│   │   │   ├── __init__.py
│   │   │   ├── dataset_utils.py
│   │   │   ├── multiturn_sft_dataset.py
│   │   │   ├── rl_dataset.py
│   │   │   ├── rm_dataset.py
│   │   │   └── vision_utils.py
│   │   ├── debug
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   ├── performance.py
│   │   │   └── trajectory_tracker.py
│   │   ├── experimental
│   │   │   ├── __init__.py
│   │   │   ├── reward_utils.py
│   │   │   └── torch_functional.py
│   │   ├── kernel
│   │   │   ├── __init__.py
│   │   │   ├── fp8_kernel.py
│   │   │   ├── kernels.py
│   │   │   └── linear_cross_entropy.py
│   │   ├── logger
│   │   │   ├── __init__.py
│   │   │   └── aggregate_logger.py
│   │   ├── megatron
│   │   │   ├── __init__.py
│   │   │   ├── dist_checkpointing.py
│   │   │   ├── memory.py
│   │   │   ├── optimizer.py
│   │   │   ├── pipeline_parallel.py
│   │   │   ├── router_replay_patch.py
│   │   │   ├── router_replay_utils.py
│   │   │   ├── sequence_parallel.py
│   │   │   └── tensor_parallel.py
│   │   ├── metric
│   │   │   ├── __init__.py
│   │   │   └── utils.py
│   │   ├── modelopt
│   │   │   ├── __init__.py
│   │   │   ├── megatron_qat_patch.py
│   │   │   ├── qat_utils.py
│   │   │   ├── qat_weight_exporter.py
│   │   │   ├── quantize.py
│   │   │   └── vllm_modelopt_patch.py
│   │   ├── profiler
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── empty_annotations.py
│   │   │   ├── mstx_profile.py
│   │   │   ├── nvtx_profile.py
│   │   │   ├── performance.py
│   │   │   ├── profile.py
│   │   │   └── torch_profile.py
│   │   ├── qat
│   │   │   ├── __init__.py
│   │   │   ├── core.py
│   │   │   ├── linear.py
│   │   │   ├── quantizer.py
│   │   │   └── vllm_patch.py
│   │   ├── rendezvous
│   │   │   ├── __init__.py
│   │   │   └── ray_backend.py
│   │   ├── reward_score
│   │   │   ├── prime_code
│   │   │   │   ├── __init__.py
│   │   │   │   ├── testing_util.py
│   │   │   │   └── utils.py
│   │   │   ├── prime_math
│   │   │   │   ├── __init__.py
│   │   │   │   ├── grader.py
│   │   │   │   └── math_normalize.py
│   │   │   ├── sandbox_fusion
│   │   │   │   ├── __init__.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── geo3k.py
│   │   │   ├── gsm8k.py
│   │   │   ├── jpeg_compressibility.py
│   │   │   ├── math_batch.py
│   │   │   ├── math_dapo.py
│   │   │   ├── math_reward.py
│   │   │   ├── math_verify.py
│   │   │   ├── rlla.py
│   │   │   └── search_r1_like_qa_em.py
│   │   ├── sglang
│   │   │   └── sglang_fp8_utils.py
│   │   ├── trtllm
│   │   │   └── trtllm_fp8_utils.py
│   │   ├── vllm
│   │   │   ├── __init__.py
│   │   │   ├── npu_vllm_patch.py
│   │   │   ├── patch.py
│   │   │   ├── utils.py
│   │   │   └── vllm_fp8_utils.py
│   │   ├── vllm_omni
│   │   │   ├── __init__.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── activation_offload.py
│   │   ├── attention_utils.py
│   │   ├── chat_template.py
│   │   ├── config.py
│   │   ├── device.py
│   │   ├── distributed.py
│   │   ├── flops_counter.py
│   │   ├── fp8_utils.py
│   │   ├── fs.py
│   │   ├── fsdp_utils.py
│   │   ├── groupwise.py
│   │   ├── hdfs_io.py
│   │   ├── import_utils.py
│   │   ├── logging_utils.py
│   │   ├── megatron_peft_utils.py
│   │   ├── megatron_utils.py
│   │   ├── memory_utils.py
│   │   ├── model.py
│   │   ├── net_utils.py
│   │   ├── npu_flash_attn_utils.py
│   │   ├── py_functional.py
│   │   ├── ray_utils.py
│   │   ├── rollout_skip.py
│   │   ├── rollout_trace.py
│   │   ├── seqlen_balancing.py
│   │   ├── tensordict_utils.py
│   │   ├── tokenizer.py
│   │   ├── torch_dtypes.py
│   │   ├── torch_functional.py
│   │   ├── tracking.py
│   │   ├── transformers_compat.py
│   │   └── ulysses.py
│   ├── workers
│   │   ├── actor
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── dp_actor.py
│   │   │   └── megatron_actor.py
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   ├── distillation.py
│   │   │   ├── engine.py
│   │   │   ├── megatron_peft.py
│   │   │   ├── model.py
│   │   │   ├── optimizer.py
│   │   │   ├── reward.py
│   │   │   └── rollout.py
│   │   ├── critic
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── dp_critic.py
│   │   │   └── megatron_critic.py
│   │   ├── engine
│   │   │   ├── automodel
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformer_impl.py
│   │   │   │   └── utils.py
│   │   │   ├── fsdp
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformer_impl.py
│   │   │   │   └── utils.py
│   │   │   ├── megatron
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformer_impl.py
│   │   │   │   └── utils.py
│   │   │   ├── mindspeed
│   │   │   │   ├── __init__.py
│   │   │   │   └── transformer_impl.py
│   │   │   ├── torchtitan
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformer_impl.py
│   │   │   │   └── utils.py
│   │   │   ├── veomni
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformer_impl.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── utils.py
│   │   ├── reward_manager
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py
│   │   │   ├── batch.py
│   │   │   ├── dapo.py
│   │   │   ├── naive.py
│   │   │   ├── prime.py
│   │   │   └── registry.py
│   │   ├── rollout
│   │   │   ├── naive
│   │   │   │   ├── __init__.py
│   │   │   │   └── naive_rollout.py
│   │   │   ├── sglang_rollout
│   │   │   │   ├── __init__.py
│   │   │   │   ├── async_sglang_server.py
│   │   │   │   ├── http_server_engine.py
│   │   │   │   ├── sglang_rollout.py
│   │   │   │   └── utils.py
│   │   │   ├── trtllm_rollout
│   │   │   │   ├── trtllm_async_server.py
│   │   │   │   ├── trtllm_rollout.py
│   │   │   │   └── trtllm_worker_extension.py
│   │   │   ├── vllm_rollout
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bucketed_weight_transfer.py
│   │   │   │   ├── utils.py
│   │   │   │   ├── vllm_async_server.py
│   │   │   │   ├── vllm_omni_async_server.py
│   │   │   │   └── vllm_rollout.py
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── hf_rollout.py
│   │   │   ├── replica.py
│   │   │   ├── schemas.py
│   │   │   ├── tokenizer.py
│   │   │   └── utils.py
│   │   ├── sharding_manager
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── fsdp_ulysses.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── losses.py
│   │   │   └── padding.py
│   │   ├── __init__.py
│   │   ├── engine_workers.py
│   │   ├── fsdp_workers.py
│   │   └── megatron_workers.py
│   ├── __init__.py
│   ├── base_config.py
│   └── protocol.py
└── setup.py
```

## Dependency Graph

```mermaid
graph LR
    m_docs__static_js_resizable_sidebar["docs/_static/js/resizable-sidebar"]
    m_docs__static_js_runllm_widget["docs/_static/js/runllm-widget"]
    m_docs_conf["docs/conf"]
    m_examples_data_preprocess_aime2024_multiturn_w_tool["examples/data_preprocess/aime2024_multiturn_w_tool"]
    m_examples_data_preprocess_dapo_multiturn_w_tool["examples/data_preprocess/dapo_multiturn_w_tool"]
    m_examples_data_preprocess_full_hh_rlhf["examples/data_preprocess/full_hh_rlhf"]
    m_examples_data_preprocess_geo3k["examples/data_preprocess/geo3k"]
    m_examples_data_preprocess_geo3k_multiturn_w_tool["examples/data_preprocess/geo3k_multiturn_w_tool"]
    m_examples_data_preprocess_gsm8k["examples/data_preprocess/gsm8k"]
    m_examples_data_preprocess_gsm8k_multiturn_sft["examples/data_preprocess/gsm8k_multiturn_sft"]
    m_examples_data_preprocess_gsm8k_multiturn_w_interaction["examples/data_preprocess/gsm8k_multiturn_w_interaction"]
    m_examples_data_preprocess_gsm8k_multiturn_w_tool["examples/data_preprocess/gsm8k_multiturn_w_tool"]
    m_examples_data_preprocess_gsm8k_tool_agent_loop["examples/data_preprocess/gsm8k_tool_agent_loop"]
    m_examples_data_preprocess_hellaswag["examples/data_preprocess/hellaswag"]
    m_examples_data_preprocess_math_dataset["examples/data_preprocess/math_dataset"]
    m_examples_data_preprocess_multiturn["examples/data_preprocess/multiturn"]
    m_examples_data_preprocess_pokemon["examples/data_preprocess/pokemon"]
    m_examples_data_preprocess_preprocess_search_r1_dataset["examples/data_preprocess/preprocess_search_r1_dataset"]
    m_examples_fapo_trainer_prepare_data["examples/fapo_trainer/prepare_data"]
    m_examples_fapo_trainer_reward_fn["examples/fapo_trainer/reward_fn"]
    m_examples_flowgrpo_trainer_reward_fn["examples/flowgrpo_trainer/reward_fn"]
    m_examples_sglang_multiturn_gsm8k_toolcall_shaping_gsm8k_toolcall_shaping["examples/sglang_multiturn/gsm8k_toolcall_shaping/gsm8k_toolcall_shaping"]
    m_examples_sglang_multiturn_search_r1_like_local_dense_retriever_download["examples/sglang_multiturn/search_r1_like/local_dense_retriever/download"]
    m_examples_sglang_multiturn_search_r1_like_local_dense_retriever_retrieval_server["examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server"]
    m_examples_split_placement_main_ppo_split["examples/split_placement/main_ppo_split"]
    m_examples_split_placement_split_monkey_patch["examples/split_placement/split_monkey_patch"]
    m_examples_tutorial_agent_loop_get_started_sandbox["examples/tutorial/agent_loop_get_started/sandbox"]
    m_examples_vllm_omni_pipeline_qwenimage["examples/vllm_omni/pipeline_qwenimage"]
    m_examples_vllm_omni_scheduling_flow_match_sde_discrete["examples/vllm_omni/scheduling_flow_match_sde_discrete"]
    m_my_recipes___init__["my_recipes/__init__"]
    m_my_recipes_genrm_remote_reward_function["my_recipes/genrm_remote/reward_function"]
    m_my_recipes_genrm_remote_reward_function_genrm["my_recipes/genrm_remote/reward_function_genrm"]
    m_my_recipes_gym_prepare_qwen_gym_parquet["my_recipes/gym/prepare_qwen_gym_parquet"]
    m_my_recipes_insturct_following___init__["my_recipes/insturct_following/__init__"]
    m_my_recipes_insturct_following_evaluation_main["my_recipes/insturct_following/evaluation_main"]
    m_my_recipes_insturct_following_instructions["my_recipes/insturct_following/instructions"]
    m_my_recipes_insturct_following_instructions_registry["my_recipes/insturct_following/instructions_registry"]
    m_my_recipes_insturct_following_instructions_util["my_recipes/insturct_following/instructions_util"]
    m_my_recipes_insturct_following_reward_function["my_recipes/insturct_following/reward_function"]
    m_my_recipes_logic_rl_reward_fn["my_recipes/logic_rl/reward_fn"]
    m_my_recipes_rllm___init__["my_recipes/rllm/__init__"]
    m_my_recipes_rllm_globals["my_recipes/rllm/globals"]
    m_my_recipes_rllm_rewards___init__["my_recipes/rllm/rewards/__init__"]
    m_my_recipes_rllm_rewards_code_reward["my_recipes/rllm/rewards/code_reward"]
    m_my_recipes_rllm_rewards_code_utils_codeforces["my_recipes/rllm/rewards/code_utils/codeforces"]
    m_my_recipes_rllm_rewards_code_utils_firejail_exec["my_recipes/rllm/rewards/code_utils/firejail_exec"]
    m_my_recipes_rllm_rewards_code_utils_humanevalplus["my_recipes/rllm/rewards/code_utils/humanevalplus"]
    m_my_recipes_rllm_rewards_code_utils_kodcode["my_recipes/rllm/rewards/code_utils/kodcode"]
    m_my_recipes_rllm_rewards_code_utils_livecodebench["my_recipes/rllm/rewards/code_utils/livecodebench"]
    m_my_recipes_rllm_rewards_code_utils_pyext2["my_recipes/rllm/rewards/code_utils/pyext2"]
    m_my_recipes_rllm_rewards_code_utils_taco["my_recipes/rllm/rewards/code_utils/taco"]
    m_my_recipes_rllm_rewards_code_utils_utils["my_recipes/rllm/rewards/code_utils/utils"]
    m_my_recipes_rllm_rewards_math_reward["my_recipes/rllm/rewards/math_reward"]
    m_my_recipes_rllm_rewards_math_utils___init__["my_recipes/rllm/rewards/math_utils/__init__"]
    m_my_recipes_rllm_rewards_math_utils_utils["my_recipes/rllm/rewards/math_utils/utils"]
    m_my_recipes_rllm_rewards_reward_types["my_recipes/rllm/rewards/reward_types"]
    m_my_recipes_rllm_rewards_rl_reward["my_recipes/rllm/rewards/rl_reward"]
    m_my_recipes_rllm_system_prompts["my_recipes/rllm/system_prompts"]
    m_my_recipes_rllm_tools___init__["my_recipes/rllm/tools/__init__"]
    m_my_recipes_rllm_tools_code_tools___init__["my_recipes/rllm/tools/code_tools/__init__"]
    m_my_recipes_rllm_tools_code_tools_code_tool["my_recipes/rllm/tools/code_tools/code_tool"]
    m_my_recipes_rllm_tools_code_tools_e2b_tool["my_recipes/rllm/tools/code_tools/e2b_tool"]
    m_my_recipes_rllm_tools_code_tools_lcb_tool["my_recipes/rllm/tools/code_tools/lcb_tool"]
    m_my_recipes_rllm_tools_code_tools_local_tool["my_recipes/rllm/tools/code_tools/local_tool"]
    m_my_recipes_rllm_tools_code_tools_together_tool["my_recipes/rllm/tools/code_tools/together_tool"]
    m_my_recipes_rllm_tools_code_tools_utils["my_recipes/rllm/tools/code_tools/utils"]
    m_my_recipes_rllm_tools_example_tool["my_recipes/rllm/tools/example_tool"]
    m_my_recipes_rllm_tools_math_tools___init__["my_recipes/rllm/tools/math_tools/__init__"]
    m_my_recipes_rllm_tools_math_tools_calculator["my_recipes/rllm/tools/math_tools/calculator"]
    m_my_recipes_rllm_tools_multi_tool["my_recipes/rllm/tools/multi_tool"]
    m_my_recipes_rllm_tools_tool_base["my_recipes/rllm/tools/tool_base"]
    m_my_recipes_rllm_tools_utils["my_recipes/rllm/tools/utils"]
    m_my_recipes_rllm_tools_web_tools___init__["my_recipes/rllm/tools/web_tools/__init__"]
    m_my_recipes_rllm_tools_web_tools_firecrawl_tool["my_recipes/rllm/tools/web_tools/firecrawl_tool"]
    m_my_recipes_rllm_tools_web_tools_gsearch_tool["my_recipes/rllm/tools/web_tools/gsearch_tool"]
    m_my_recipes_rllm_tools_web_tools_tavily_tool["my_recipes/rllm/tools/web_tools/tavily_tool"]
    m_my_recipes_rllm_utils["my_recipes/rllm/utils"]
    m_my_recipes_syntool___init__["my_recipes/syntool/__init__"]
    m_my_recipes_syntool_agent_loop["my_recipes/syntool/agent_loop"]
    m_my_recipes_syntool_dataset["my_recipes/syntool/dataset"]
    m_my_recipes_syntool_reward["my_recipes/syntool/reward"]
    m_my_recipes_syntool_tool["my_recipes/syntool/tool"]
    m_scripts___init__["scripts/__init__"]
    m_scripts_converter_hf_to_mcore["scripts/converter_hf_to_mcore"]
    m_scripts_diagnose["scripts/diagnose"]
    m_scripts_init_random_model["scripts/init_random_model"]
    m_scripts_legacy_model_merger["scripts/legacy_model_merger"]
    m_scripts_megatron_merge_lora["scripts/megatron_merge_lora"]
    m_scripts_print_cfg["scripts/print_cfg"]
    m_scripts_rollout_viewer["scripts/rollout_viewer"]
    m_scripts_veomni_moe_merge["scripts/veomni/moe_merge"]
    m_scripts_veomni_moe_split["scripts/veomni/moe_split"]
    m_setup["setup"]
    m_tests___init__["tests/__init__"]
    m_tests_checkpoint_engine___init__["tests/checkpoint_engine/__init__"]
    m_tests_checkpoint_engine_test_correctness_on_gpu["tests/checkpoint_engine/test_correctness_on_gpu"]
    m_tests_checkpoint_engine_test_correctness_on_npu["tests/checkpoint_engine/test_correctness_on_npu"]
    m_tests_checkpoint_engine_test_special_server_adapter["tests/checkpoint_engine/test_special_server_adapter"]
    m_tests_checkpoint_engine_test_utils["tests/checkpoint_engine/test_utils"]
    m_tests_experimental_agent_loop_agent_utils["tests/experimental/agent_loop/agent_utils"]
    m_tests_experimental_agent_loop_test_agent_loop_extra_fields_schema_on_cpu["tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu"]
    m_tests_experimental_agent_loop_test_basic_agent_loop["tests/experimental/agent_loop/test_basic_agent_loop"]
    m_tests_experimental_agent_loop_test_gpt_oss_tool_parser["tests/experimental/agent_loop/test_gpt_oss_tool_parser"]
    m_tests_experimental_agent_loop_test_multi_modal["tests/experimental/agent_loop/test_multi_modal"]
    m_tests_experimental_agent_loop_test_standalone_rollout["tests/experimental/agent_loop/test_standalone_rollout"]
    m_tests_experimental_reward_loop_reward_fn["tests/experimental/reward_loop/reward_fn"]
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate["tests/experimental/reward_loop/test_agent_reward_loop_colocate"]
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone["tests/experimental/reward_loop/test_agent_reward_loop_standalone"]
    m_tests_experimental_reward_loop_test_async_token_bucket_on_cpu["tests/experimental/reward_loop/test_async_token_bucket_on_cpu"]
    m_tests_experimental_reward_loop_test_math_verify["tests/experimental/reward_loop/test_math_verify"]
    m_tests_experimental_reward_loop_test_rate_limited_reward_manager_on_cpu["tests/experimental/reward_loop/test_rate_limited_reward_manager_on_cpu"]
    m_tests_experimental_reward_loop_test_reward_model_disrm["tests/experimental/reward_loop/test_reward_model_disrm"]
    m_tests_experimental_reward_loop_test_reward_model_genrm["tests/experimental/reward_loop/test_reward_model_genrm"]
    m_tests_experimental_reward_loop_test_visual_reward_manager["tests/experimental/reward_loop/test_visual_reward_manager"]
    m_tests_experimental_vla_test_sim_envs["tests/experimental/vla/test_sim_envs"]
    m_tests_interactions___init__["tests/interactions/__init__"]
    m_tests_interactions_test_gsm8k_interaction["tests/interactions/test_gsm8k_interaction"]
    m_tests_interactions_test_interaction_registry["tests/interactions/test_interaction_registry"]
    m_tests_models_test_engine["tests/models/test_engine"]
    m_tests_models_test_liger_vl_compat["tests/models/test_liger_vl_compat"]
    m_tests_models_test_tiled_mlp_accuracy["tests/models/test_tiled_mlp_accuracy"]
    m_tests_models_test_transformer["tests/models/test_transformer"]
    m_tests_models_test_transformers_ulysses["tests/models/test_transformers_ulysses"]
    m_tests_my_recipes_test_syntool_recipe["tests/my_recipes/test_syntool_recipe"]
    m_tests_single_controller___init__["tests/single_controller/__init__"]
    m_tests_single_controller_base_test_decorator["tests/single_controller/base/test_decorator"]
    m_tests_single_controller_check_worker_alive_main["tests/single_controller/check_worker_alive/main"]
    m_tests_single_controller_detached_worker_client["tests/single_controller/detached_worker/client"]
    m_tests_single_controller_detached_worker_server["tests/single_controller/detached_worker/server"]
    m_tests_single_controller_test_auto_padding_on_cpu["tests/single_controller/test_auto_padding_on_cpu"]
    m_tests_single_controller_test_colocated_workers["tests/single_controller/test_colocated_workers"]
    m_tests_single_controller_test_colocated_workers_fused["tests/single_controller/test_colocated_workers_fused"]
    m_tests_single_controller_test_data_transfer["tests/single_controller/test_data_transfer"]
    m_tests_single_controller_test_decorator_on_cpu["tests/single_controller/test_decorator_on_cpu"]
    m_tests_single_controller_test_device_mesh_register["tests/single_controller/test_device_mesh_register"]
    m_tests_single_controller_test_driverfunc_to_worker["tests/single_controller/test_driverfunc_to_worker"]
    m_tests_single_controller_test_fused_workers_on_cpu["tests/single_controller/test_fused_workers_on_cpu"]
    m_tests_single_controller_test_get_set_dispatch_collect_cpu["tests/single_controller/test_get_set_dispatch_collect_cpu"]
    m_tests_single_controller_test_high_level_scheduling_api["tests/single_controller/test_high_level_scheduling_api"]
    m_tests_single_controller_test_nested_worker["tests/single_controller/test_nested_worker"]
    m_tests_single_controller_test_ray_collectives["tests/single_controller/test_ray_collectives"]
    m_tests_single_controller_test_ray_local_envs_on_cpu["tests/single_controller/test_ray_local_envs_on_cpu"]
    m_tests_single_controller_test_ray_utils_on_cpu["tests/single_controller/test_ray_utils_on_cpu"]
    m_tests_single_controller_test_rvdz["tests/single_controller/test_rvdz"]
    m_tests_single_controller_test_split_resource_pool["tests/single_controller/test_split_resource_pool"]
    m_tests_single_controller_test_worker_group_basics["tests/single_controller/test_worker_group_basics"]
    m_tests_single_controller_test_worker_group_torch["tests/single_controller/test_worker_group_torch"]
    m_tests_special_distributed_test_fsdp_ckpt["tests/special_distributed/test_fsdp_ckpt"]
    m_tests_special_distributed_test_mcore_config_converter["tests/special_distributed/test_mcore_config_converter"]
    m_tests_special_distributed_test_tensor_dict["tests/special_distributed/test_tensor_dict"]
    m_tests_special_distributed_test_torch_functional["tests/special_distributed/test_torch_functional"]
    m_tests_special_e2e___init__["tests/special_e2e/__init__"]
    m_tests_special_e2e_check_custom_rwd_fn["tests/special_e2e/check_custom_rwd_fn"]
    m_tests_special_e2e_check_results["tests/special_e2e/check_results"]
    m_tests_special_e2e_envs___init__["tests/special_e2e/envs/__init__"]
    m_tests_special_e2e_envs_digit_completion___init__["tests/special_e2e/envs/digit_completion/__init__"]
    m_tests_special_e2e_envs_digit_completion_task["tests/special_e2e/envs/digit_completion/task"]
    m_tests_special_e2e_envs_digit_completion_tokenizer["tests/special_e2e/envs/digit_completion/tokenizer"]
    m_tests_special_e2e_sft_compare_sft_engine_results["tests/special_e2e/sft/compare_sft_engine_results"]
    m_tests_special_sanity_check_api_docs["tests/special_sanity/check_api_docs"]
    m_tests_special_sanity_check_dataproto_usage["tests/special_sanity/check_dataproto_usage"]
    m_tests_special_sanity_check_device_api_usage["tests/special_sanity/check_device_api_usage"]
    m_tests_special_sanity_check_docs_time_info["tests/special_sanity/check_docs_time_info"]
    m_tests_special_sanity_check_docstrings["tests/special_sanity/check_docstrings"]
    m_tests_special_sanity_check_license["tests/special_sanity/check_license"]
    m_tests_special_sanity_check_pr_description["tests/special_sanity/check_pr_description"]
    m_tests_special_sanity_check_pr_title["tests/special_sanity/check_pr_title"]
    m_tests_special_sanity_test_config_docs["tests/special_sanity/test_config_docs"]
    m_tests_special_sanity_test_import["tests/special_sanity/test_import"]
    m_tests_special_sanity_type_coverage_check["tests/special_sanity/type_coverage_check"]
    m_tests_special_sanity_validate_imported_docs["tests/special_sanity/validate_imported_docs"]
    m_tests_special_sanity_validate_structure["tests/special_sanity/validate_structure"]
    m_tests_special_standalone_test_memory_buffers["tests/special_standalone/test_memory_buffers"]
    m_tests_test_base_config_on_cpu["tests/test_base_config_on_cpu"]
    m_tests_test_protocol_on_cpu["tests/test_protocol_on_cpu"]
    m_tests_test_protocol_v2_on_cpu["tests/test_protocol_v2_on_cpu"]
    m_tests_trainer___init__["tests/trainer/__init__"]
    m_tests_trainer_config___init__["tests/trainer/config/__init__"]
    m_tests_trainer_config_test_algo_config_on_cpu["tests/trainer/config/test_algo_config_on_cpu"]
    m_tests_trainer_config_test_legacy_config_on_cpu["tests/trainer/config/test_legacy_config_on_cpu"]
    m_tests_trainer_ppo___init__["tests/trainer/ppo/__init__"]
    m_tests_trainer_ppo_test_core_algos_on_cpu["tests/trainer/ppo/test_core_algos_on_cpu"]
    m_tests_trainer_ppo_test_metric_utils_on_cpu["tests/trainer/ppo/test_metric_utils_on_cpu"]
    m_tests_trainer_ppo_test_rollout_corr["tests/trainer/ppo/test_rollout_corr"]
    m_tests_trainer_ppo_test_rollout_corr_integration["tests/trainer/ppo/test_rollout_corr_integration"]
    m_tests_utils__test_module["tests/utils/_test_module"]
    m_tests_utils_ckpt_test_checkpoint_cleanup_on_cpu["tests/utils/ckpt/test_checkpoint_cleanup_on_cpu"]
    m_tests_utils_ckpt_test_esi_save_ckpt_on_cpu["tests/utils/ckpt/test_esi_save_ckpt_on_cpu"]
    m_tests_utils_dataset_test_create_rl_sampler_on_cpu["tests/utils/dataset/test_create_rl_sampler_on_cpu"]
    m_tests_utils_dataset_test_multiturn_sft_dataset_on_cpu["tests/utils/dataset/test_multiturn_sft_dataset_on_cpu"]
    m_tests_utils_dataset_test_rl_collate_fn_on_cpu["tests/utils/dataset/test_rl_collate_fn_on_cpu"]
    m_tests_utils_dataset_test_rl_dataset_on_cpu["tests/utils/dataset/test_rl_dataset_on_cpu"]
    m_tests_utils_debug_test_metrics["tests/utils/debug/test_metrics"]
    m_tests_utils_megatron_test_pipeline_parallel["tests/utils/megatron/test_pipeline_parallel"]
    m_tests_utils_reward_score_reward_score_test_sandbox_fusion_on_cpu["tests/utils/reward_score/reward_score/test_sandbox_fusion_on_cpu"]
    m_tests_utils_reward_score_test_sandbox_on_cpu["tests/utils/reward_score/test_sandbox_on_cpu"]
    m_tests_utils_test_activation_offload["tests/utils/test_activation_offload"]
    m_tests_utils_test_bucketed_weight_transfer["tests/utils/test_bucketed_weight_transfer"]
    m_tests_utils_test_check_ipc_version_support_on_npu["tests/utils/test_check_ipc_version_support_on_npu"]
    m_tests_utils_test_check_profiler_output["tests/utils/test_check_profiler_output"]
    m_tests_utils_test_config_on_cpu["tests/utils/test_config_on_cpu"]
    m_tests_utils_test_flops_counter["tests/utils/test_flops_counter"]
    m_tests_utils_test_fs_on_cpu["tests/utils/test_fs_on_cpu"]
    m_tests_utils_test_fsdp2_peft_wrapping["tests/utils/test_fsdp2_peft_wrapping"]
    m_tests_utils_test_fsdp_lora_merge["tests/utils/test_fsdp_lora_merge"]
    m_tests_utils_test_groupwise["tests/utils/test_groupwise"]
    m_tests_utils_test_import_utils_on_cpu["tests/utils/test_import_utils_on_cpu"]
    m_tests_utils_test_linear_cross_entropy["tests/utils/test_linear_cross_entropy"]
    m_tests_utils_test_mlflow_key_sanitization["tests/utils/test_mlflow_key_sanitization"]
    m_tests_utils_test_model_on_cpu["tests/utils/test_model_on_cpu"]
    m_tests_utils_test_normalize_peft_param_name["tests/utils/test_normalize_peft_param_name"]
    m_tests_utils_test_normalize_peft_param_name_on_cpu["tests/utils/test_normalize_peft_param_name_on_cpu"]
    m_tests_utils_test_nvtx_profile["tests/utils/test_nvtx_profile"]
    m_tests_utils_test_padding_on_cpu["tests/utils/test_padding_on_cpu"]
    m_tests_utils_test_prepare_micro_batches_with_group_size["tests/utils/test_prepare_micro_batches_with_group_size"]
    m_tests_utils_test_rollout_skip_on_cpu["tests/utils/test_rollout_skip_on_cpu"]
    m_tests_utils_test_rollout_trace_on_cpu["tests/utils/test_rollout_trace_on_cpu"]
    m_tests_utils_test_seqlen_balancing["tests/utils/test_seqlen_balancing"]
    m_tests_utils_test_server_profiler["tests/utils/test_server_profiler"]
    m_tests_utils_test_shared_memory["tests/utils/test_shared_memory"]
    m_tests_utils_test_special_linear_cross_entropy_tp["tests/utils/test_special_linear_cross_entropy_tp"]
    m_tests_utils_test_special_megatron_kl_loss_tp["tests/utils/test_special_megatron_kl_loss_tp"]
    m_tests_utils_test_special_mstx_profile["tests/utils/test_special_mstx_profile"]
    m_tests_utils_test_temp_env_on_cpu["tests/utils/test_temp_env_on_cpu"]
    m_tests_utils_test_timeout_decorator_cpu["tests/utils/test_timeout_decorator_cpu"]
    m_tests_utils_test_tokenizer_normalize_on_cpu["tests/utils/test_tokenizer_normalize_on_cpu"]
    m_tests_utils_test_torch_functional["tests/utils/test_torch_functional"]
    m_tests_utils_test_torch_profile["tests/utils/test_torch_profile"]
    m_tests_workers_actor_test_special_dp_actor["tests/workers/actor/test_special_dp_actor"]
    m_tests_workers_config_test_actor_config_on_cpu["tests/workers/config/test_actor_config_on_cpu"]
    m_tests_workers_config_test_critic_config_on_cpu["tests/workers/config/test_critic_config_on_cpu"]
    m_tests_workers_config_test_engine_config_on_cpu["tests/workers/config/test_engine_config_on_cpu"]
    m_tests_workers_config_test_model_config_on_cpu["tests/workers/config/test_model_config_on_cpu"]
    m_tests_workers_config_test_optim_config_on_cpu["tests/workers/config/test_optim_config_on_cpu"]
    m_tests_workers_critic_test_special_dp_critic["tests/workers/critic/test_special_dp_critic"]
    m_tests_workers_reward_manager_test_registry_on_cpu["tests/workers/reward_manager/test_registry_on_cpu"]
    m_tests_workers_rollout_perf_vllm_async_rollout["tests/workers/rollout/perf/vllm_async_rollout"]
    m_tests_workers_rollout_rollout_sglang_test_http_server_engine["tests/workers/rollout/rollout_sglang/test_http_server_engine"]
    m_tests_workers_rollout_rollout_sglang_test_lora_sleep_level["tests/workers/rollout/rollout_sglang/test_lora_sleep_level"]
    m_tests_workers_rollout_rollout_trtllm___init__["tests/workers/rollout/rollout_trtllm/__init__"]
    m_tests_workers_rollout_rollout_trtllm_test_adapter["tests/workers/rollout/rollout_trtllm/test_adapter"]
    m_tests_workers_rollout_rollout_trtllm_test_async_server["tests/workers/rollout/rollout_trtllm/test_async_server"]
    m_tests_workers_rollout_rollout_trtllm_test_trtllm_rollout_utils["tests/workers/rollout/rollout_trtllm/test_trtllm_rollout_utils"]
    m_tests_workers_rollout_rollout_vllm_run_fsdp_vllm["tests/workers/rollout/rollout_vllm/run_fsdp_vllm"]
    m_tests_workers_rollout_rollout_vllm_test_vllm_abort["tests/workers/rollout/rollout_vllm/test_vllm_abort"]
    m_tests_workers_rollout_rollout_vllm_test_vllm_omni_generate["tests/workers/rollout/rollout_vllm/test_vllm_omni_generate"]
    m_tests_workers_rollout_test_hf_rollout["tests/workers/rollout/test_hf_rollout"]
    m_tests_workers_rollout_test_sglang_async_rollout_multimodal_delta["tests/workers/rollout/test_sglang_async_rollout_multimodal_delta"]
    m_tests_workers_rollout_test_sglang_rollout_sharding_manager["tests/workers/rollout/test_sglang_rollout_sharding_manager"]
    m_tests_workers_rollout_test_vllm_cli_args_on_cpu["tests/workers/rollout/test_vllm_cli_args_on_cpu"]
    m_tests_workers_test_fsdp_attn_implementation["tests/workers/test_fsdp_attn_implementation"]
    m_tests_workers_test_fsdp_workers["tests/workers/test_fsdp_workers"]
    m_verl___init__["verl/__init__"]
    m_verl_base_config["verl/base_config"]
    m_verl_checkpoint_engine___init__["verl/checkpoint_engine/__init__"]
    m_verl_checkpoint_engine_base["verl/checkpoint_engine/base"]
    m_verl_checkpoint_engine_hccl_checkpoint_engine["verl/checkpoint_engine/hccl_checkpoint_engine"]
    m_verl_checkpoint_engine_kimi_checkpoint_engine["verl/checkpoint_engine/kimi_checkpoint_engine"]
    m_verl_checkpoint_engine_mooncake_checkpoint_engine["verl/checkpoint_engine/mooncake_checkpoint_engine"]
    m_verl_checkpoint_engine_nccl_checkpoint_engine["verl/checkpoint_engine/nccl_checkpoint_engine"]
    m_verl_checkpoint_engine_nixl_checkpoint_engine["verl/checkpoint_engine/nixl_checkpoint_engine"]
    m_verl_experimental___init__["verl/experimental/__init__"]
    m_verl_experimental_agent_loop___init__["verl/experimental/agent_loop/__init__"]
    m_verl_experimental_agent_loop_agent_loop["verl/experimental/agent_loop/agent_loop"]
    m_verl_experimental_agent_loop_prometheus_utils["verl/experimental/agent_loop/prometheus_utils"]
    m_verl_experimental_agent_loop_single_turn_agent_loop["verl/experimental/agent_loop/single_turn_agent_loop"]
    m_verl_experimental_agent_loop_tool_agent_loop["verl/experimental/agent_loop/tool_agent_loop"]
    m_verl_experimental_agent_loop_tool_parser["verl/experimental/agent_loop/tool_parser"]
    m_verl_experimental_agent_loop_utils["verl/experimental/agent_loop/utils"]
    m_verl_experimental_dataset___init__["verl/experimental/dataset/__init__"]
    m_verl_experimental_dataset_sampler["verl/experimental/dataset/sampler"]
    m_verl_experimental_dynamic_dataset___init__["verl/experimental/dynamic_dataset/__init__"]
    m_verl_experimental_dynamic_dataset_dynamicgen_dataset["verl/experimental/dynamic_dataset/dynamicgen_dataset"]
    m_verl_experimental_fully_async_policy_agent_loop___init__["verl/experimental/fully_async_policy/agent_loop/__init__"]
    m_verl_experimental_fully_async_policy_agent_loop_agent_loop["verl/experimental/fully_async_policy/agent_loop/agent_loop"]
    m_verl_experimental_fully_async_policy_detach_utils["verl/experimental/fully_async_policy/detach_utils"]
    m_verl_experimental_fully_async_policy_fully_async_main["verl/experimental/fully_async_policy/fully_async_main"]
    m_verl_experimental_fully_async_policy_fully_async_rollouter["verl/experimental/fully_async_policy/fully_async_rollouter"]
    m_verl_experimental_fully_async_policy_fully_async_trainer["verl/experimental/fully_async_policy/fully_async_trainer"]
    m_verl_experimental_fully_async_policy_message_queue["verl/experimental/fully_async_policy/message_queue"]
    m_verl_experimental_fully_async_policy_unittest_simple_streaming_demo["verl/experimental/fully_async_policy/unittest/simple_streaming_demo"]
    m_verl_experimental_one_step_off_policy_main_ppo["verl/experimental/one_step_off_policy/main_ppo"]
    m_verl_experimental_one_step_off_policy_ray_trainer["verl/experimental/one_step_off_policy/ray_trainer"]
    m_verl_experimental_reward_loop___init__["verl/experimental/reward_loop/__init__"]
    m_verl_experimental_reward_loop_reward_loop["verl/experimental/reward_loop/reward_loop"]
    m_verl_experimental_reward_loop_reward_manager___init__["verl/experimental/reward_loop/reward_manager/__init__"]
    m_verl_experimental_reward_loop_reward_manager_base["verl/experimental/reward_loop/reward_manager/base"]
    m_verl_experimental_reward_loop_reward_manager_dapo["verl/experimental/reward_loop/reward_manager/dapo"]
    m_verl_experimental_reward_loop_reward_manager_gdpo["verl/experimental/reward_loop/reward_manager/gdpo"]
    m_verl_experimental_reward_loop_reward_manager_limited["verl/experimental/reward_loop/reward_manager/limited"]
    m_verl_experimental_reward_loop_reward_manager_naive["verl/experimental/reward_loop/reward_manager/naive"]
    m_verl_experimental_reward_loop_reward_manager_registry["verl/experimental/reward_loop/reward_manager/registry"]
    m_verl_experimental_reward_loop_reward_manager_remote["verl/experimental/reward_loop/reward_manager/remote"]
    m_verl_experimental_reward_loop_reward_manager_visual["verl/experimental/reward_loop/reward_manager/visual"]
    m_verl_experimental_reward_loop_reward_model["verl/experimental/reward_loop/reward_model"]
    m_verl_experimental_reward_loop_router_inner_sglang_router["verl/experimental/reward_loop/router/inner_sglang_router"]
    m_verl_experimental_reward_loop_router_naive_router["verl/experimental/reward_loop/router/naive_router"]
    m_verl_experimental_separation___init__["verl/experimental/separation/__init__"]
    m_verl_experimental_separation_engine_workers["verl/experimental/separation/engine_workers"]
    m_verl_experimental_separation_ray_trainer["verl/experimental/separation/ray_trainer"]
    m_verl_experimental_separation_utils["verl/experimental/separation/utils"]
    m_verl_experimental_teacher_loop___init__["verl/experimental/teacher_loop/__init__"]
    m_verl_experimental_teacher_loop_teacher_manager["verl/experimental/teacher_loop/teacher_manager"]
    m_verl_experimental_teacher_loop_teacher_model["verl/experimental/teacher_loop/teacher_model"]
    m_verl_experimental_vla_dp_rob["verl/experimental/vla/dp_rob"]
    m_verl_experimental_vla_env_loop["verl/experimental/vla/env_loop"]
    m_verl_experimental_vla_envs___init__["verl/experimental/vla/envs/__init__"]
    m_verl_experimental_vla_envs_action_utils["verl/experimental/vla/envs/action_utils"]
    m_verl_experimental_vla_envs_isaac_env___init__["verl/experimental/vla/envs/isaac_env/__init__"]
    m_verl_experimental_vla_envs_isaac_env_isaac_env["verl/experimental/vla/envs/isaac_env/isaac_env"]
    m_verl_experimental_vla_envs_libero_env___init__["verl/experimental/vla/envs/libero_env/__init__"]
    m_verl_experimental_vla_envs_libero_env_libero_env["verl/experimental/vla/envs/libero_env/libero_env"]
    m_verl_experimental_vla_envs_libero_env_utils["verl/experimental/vla/envs/libero_env/utils"]
    m_verl_experimental_vla_envs_libero_env_venv["verl/experimental/vla/envs/libero_env/venv"]
    m_verl_experimental_vla_fsdp_workers["verl/experimental/vla/fsdp_workers"]
    m_verl_experimental_vla_main_ppo["verl/experimental/vla/main_ppo"]
    m_verl_experimental_vla_main_sac["verl/experimental/vla/main_sac"]
    m_verl_experimental_vla_models___init__["verl/experimental/vla/models/__init__"]
    m_verl_experimental_vla_models_modules_mlp["verl/experimental/vla/models/modules/mlp"]
    m_verl_experimental_vla_models_openvla_oft___init__["verl/experimental/vla/models/openvla_oft/__init__"]
    m_verl_experimental_vla_models_openvla_oft_configuration_prismatic["verl/experimental/vla/models/openvla_oft/configuration_prismatic"]
    m_verl_experimental_vla_models_openvla_oft_constants["verl/experimental/vla/models/openvla_oft/constants"]
    m_verl_experimental_vla_models_openvla_oft_modeling_prismatic["verl/experimental/vla/models/openvla_oft/modeling_prismatic"]
    m_verl_experimental_vla_models_openvla_oft_processing_prismatic["verl/experimental/vla/models/openvla_oft/processing_prismatic"]
    m_verl_experimental_vla_models_openvla_oft_train_utils["verl/experimental/vla/models/openvla_oft/train_utils"]
    m_verl_experimental_vla_models_pi0_torch___init__["verl/experimental/vla/models/pi0_torch/__init__"]
    m_verl_experimental_vla_models_pi0_torch_configuration_pi0_torch["verl/experimental/vla/models/pi0_torch/configuration_pi0_torch"]
    m_verl_experimental_vla_models_pi0_torch_model_modeling_pi0["verl/experimental/vla/models/pi0_torch/model/modeling_pi0"]
    m_verl_experimental_vla_models_pi0_torch_model_paligemma_with_expert["verl/experimental/vla/models/pi0_torch/model/paligemma_with_expert"]
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch["verl/experimental/vla/models/pi0_torch/modeling_pi0_torch"]
    m_verl_experimental_vla_models_pi0_torch_pi0_utils["verl/experimental/vla/models/pi0_torch/pi0_utils"]
    m_verl_experimental_vla_models_pi0_torch_policy___init__["verl/experimental/vla/models/pi0_torch/policy/__init__"]
    m_verl_experimental_vla_models_pi0_torch_policy_base["verl/experimental/vla/models/pi0_torch/policy/base"]
    m_verl_experimental_vla_models_pi0_torch_policy_libero_policy["verl/experimental/vla/models/pi0_torch/policy/libero_policy"]
    m_verl_experimental_vla_models_register_vla_models["verl/experimental/vla/models/register_vla_models"]
    m_verl_experimental_vla_naive_rollout_rob["verl/experimental/vla/naive_rollout_rob"]
    m_verl_experimental_vla_prepare_libero_dataset["verl/experimental/vla/prepare_libero_dataset"]
    m_verl_experimental_vla_rob_ray_trainer["verl/experimental/vla/rob_ray_trainer"]
    m_verl_experimental_vla_sac_base["verl/experimental/vla/sac/base"]
    m_verl_experimental_vla_sac_naive_rollout_pi05["verl/experimental/vla/sac/naive_rollout_pi05"]
    m_verl_experimental_vla_sac_replay_pool["verl/experimental/vla/sac/replay_pool"]
    m_verl_experimental_vla_sac_sac_actor["verl/experimental/vla/sac/sac_actor"]
    m_verl_experimental_vla_sac_sac_ray_trainer["verl/experimental/vla/sac/sac_ray_trainer"]
    m_verl_experimental_vla_workers_env_env_loop_wg_test["verl/experimental/vla/workers/env/env_loop_wg_test"]
    m_verl_experimental_vla_workers_env_env_manager["verl/experimental/vla/workers/env/env_manager"]
    m_verl_experimental_vla_workers_env_env_worker["verl/experimental/vla/workers/env/env_worker"]
    m_verl_interactions___init__["verl/interactions/__init__"]
    m_verl_interactions_base["verl/interactions/base"]
    m_verl_interactions_gsm8k_interaction["verl/interactions/gsm8k_interaction"]
    m_verl_interactions_gym_env["verl/interactions/gym_env"]
    m_verl_interactions_gym_interaction["verl/interactions/gym_interaction"]
    m_verl_interactions_utils___init__["verl/interactions/utils/__init__"]
    m_verl_interactions_utils_interaction_registry["verl/interactions/utils/interaction_registry"]
    m_verl_interactions_weather_interaction["verl/interactions/weather_interaction"]
    m_verl_model_merger___init__["verl/model_merger/__init__"]
    m_verl_model_merger___main__["verl/model_merger/__main__"]
    m_verl_model_merger_base_model_merger["verl/model_merger/base_model_merger"]
    m_verl_model_merger_fsdp_model_merger["verl/model_merger/fsdp_model_merger"]
    m_verl_model_merger_megatron_model_merger["verl/model_merger/megatron_model_merger"]
    m_verl_models___init__["verl/models/__init__"]
    m_verl_models_mcore___init__["verl/models/mcore/__init__"]
    m_verl_models_mcore_bridge["verl/models/mcore/bridge"]
    m_verl_models_mcore_config_converter["verl/models/mcore/config_converter"]
    m_verl_models_mcore_loader["verl/models/mcore/loader"]
    m_verl_models_mcore_mbridge["verl/models/mcore/mbridge"]
    m_verl_models_mcore_model_forward["verl/models/mcore/model_forward"]
    m_verl_models_mcore_model_forward_1f1b_overlap["verl/models/mcore/model_forward_1f1b_overlap"]
    m_verl_models_mcore_model_forward_fused["verl/models/mcore/model_forward_fused"]
    m_verl_models_mcore_model_initializer["verl/models/mcore/model_initializer"]
    m_verl_models_mcore_mtp_patch["verl/models/mcore/mtp_patch"]
    m_verl_models_mcore_patch["verl/models/mcore/patch"]
    m_verl_models_mcore_qwen2_5_vl___init__["verl/models/mcore/qwen2_5_vl/__init__"]
    m_verl_models_mcore_qwen2_5_vl_attention["verl/models/mcore/qwen2_5_vl/attention"]
    m_verl_models_mcore_qwen2_5_vl_model["verl/models/mcore/qwen2_5_vl/model"]
    m_verl_models_mcore_qwen2_5_vl_rope_utils["verl/models/mcore/qwen2_5_vl/rope_utils"]
    m_verl_models_mcore_qwen2_5_vl_vision_config["verl/models/mcore/qwen2_5_vl/vision_config"]
    m_verl_models_mcore_qwen2_5_vl_vision_model["verl/models/mcore/qwen2_5_vl/vision_model"]
    m_verl_models_mcore_qwen2_5_vl_vision_transformer_block["verl/models/mcore/qwen2_5_vl/vision_transformer_block"]
    m_verl_models_mcore_registry["verl/models/mcore/registry"]
    m_verl_models_mcore_saver["verl/models/mcore/saver"]
    m_verl_models_mcore_util["verl/models/mcore/util"]
    m_verl_models_mcore_weight_converter["verl/models/mcore/weight_converter"]
    m_verl_models_registry["verl/models/registry"]
    m_verl_models_transformers___init__["verl/models/transformers/__init__"]
    m_verl_models_transformers_apertus["verl/models/transformers/apertus"]
    m_verl_models_transformers_dense_common["verl/models/transformers/dense_common"]
    m_verl_models_transformers_glm4v["verl/models/transformers/glm4v"]
    m_verl_models_transformers_kimi_vl["verl/models/transformers/kimi_vl"]
    m_verl_models_transformers_llama["verl/models/transformers/llama"]
    m_verl_models_transformers_monkey_patch["verl/models/transformers/monkey_patch"]
    m_verl_models_transformers_npu_patch["verl/models/transformers/npu_patch"]
    m_verl_models_transformers_qwen2["verl/models/transformers/qwen2"]
    m_verl_models_transformers_qwen2_vl["verl/models/transformers/qwen2_vl"]
    m_verl_models_transformers_qwen3_vl["verl/models/transformers/qwen3_vl"]
    m_verl_models_transformers_tiled_mlp["verl/models/transformers/tiled_mlp"]
    m_verl_models_weight_loader_registry["verl/models/weight_loader_registry"]
    m_verl_protocol["verl/protocol"]
    m_verl_single_controller___init__["verl/single_controller/__init__"]
    m_verl_single_controller_base___init__["verl/single_controller/base/__init__"]
    m_verl_single_controller_base_decorator["verl/single_controller/base/decorator"]
    m_verl_single_controller_base_worker["verl/single_controller/base/worker"]
    m_verl_single_controller_base_worker_group["verl/single_controller/base/worker_group"]
    m_verl_single_controller_ray___init__["verl/single_controller/ray/__init__"]
    m_verl_single_controller_ray_base["verl/single_controller/ray/base"]
    m_verl_third_party___init__["verl/third_party/__init__"]
    m_verl_third_party_torch___init__["verl/third_party/torch/__init__"]
    m_verl_third_party_torch_distributed___init__["verl/third_party/torch/distributed/__init__"]
    m_verl_third_party_torch_distributed__state_dict_utils["verl/third_party/torch/distributed/_state_dict_utils"]
    m_verl_third_party_torch_distributed_checkpoint___init__["verl/third_party/torch/distributed/checkpoint/__init__"]
    m_verl_third_party_torch_distributed_checkpoint_state_dict["verl/third_party/torch/distributed/checkpoint/state_dict"]
    m_verl_third_party_vllm___init__["verl/third_party/vllm/__init__"]
    m_verl_tools___init__["verl/tools/__init__"]
    m_verl_tools_base_tool["verl/tools/base_tool"]
    m_verl_tools_geo3k_tool["verl/tools/geo3k_tool"]
    m_verl_tools_gsm8k_tool["verl/tools/gsm8k_tool"]
    m_verl_tools_image_zoom_in_tool["verl/tools/image_zoom_in_tool"]
    m_verl_tools_mcp_base_tool["verl/tools/mcp_base_tool"]
    m_verl_tools_mcp_search_tool["verl/tools/mcp_search_tool"]
    m_verl_tools_sandbox_fusion_tools["verl/tools/sandbox_fusion_tools"]
    m_verl_tools_schemas["verl/tools/schemas"]
    m_verl_tools_search_tool["verl/tools/search_tool"]
    m_verl_tools_utils___init__["verl/tools/utils/__init__"]
    m_verl_tools_utils_mcp_clients_McpClientManager["verl/tools/utils/mcp_clients/McpClientManager"]
    m_verl_tools_utils_mcp_clients_utils["verl/tools/utils/mcp_clients/utils"]
    m_verl_tools_utils_search_r1_like_utils["verl/tools/utils/search_r1_like_utils"]
    m_verl_tools_utils_tool_registry["verl/tools/utils/tool_registry"]
    m_verl_trainer___init__["verl/trainer/__init__"]
    m_verl_trainer_config___init__["verl/trainer/config/__init__"]
    m_verl_trainer_config_algorithm["verl/trainer/config/algorithm"]
    m_verl_trainer_config_config["verl/trainer/config/config"]
    m_verl_trainer_constants_ppo["verl/trainer/constants_ppo"]
    m_verl_trainer_distillation___init__["verl/trainer/distillation/__init__"]
    m_verl_trainer_distillation_fsdp_losses["verl/trainer/distillation/fsdp/losses"]
    m_verl_trainer_distillation_losses["verl/trainer/distillation/losses"]
    m_verl_trainer_distillation_megatron_losses["verl/trainer/distillation/megatron/losses"]
    m_verl_trainer_main_eval["verl/trainer/main_eval"]
    m_verl_trainer_main_generation_server["verl/trainer/main_generation_server"]
    m_verl_trainer_main_ppo["verl/trainer/main_ppo"]
    m_verl_trainer_ppo___init__["verl/trainer/ppo/__init__"]
    m_verl_trainer_ppo_core_algos["verl/trainer/ppo/core_algos"]
    m_verl_trainer_ppo_metric_utils["verl/trainer/ppo/metric_utils"]
    m_verl_trainer_ppo_prefix_grouper_utils["verl/trainer/ppo/prefix_grouper_utils"]
    m_verl_trainer_ppo_ray_trainer["verl/trainer/ppo/ray_trainer"]
    m_verl_trainer_ppo_reward["verl/trainer/ppo/reward"]
    m_verl_trainer_ppo_rollout_corr_helper["verl/trainer/ppo/rollout_corr_helper"]
    m_verl_trainer_ppo_utils["verl/trainer/ppo/utils"]
    m_verl_trainer_sft_trainer["verl/trainer/sft_trainer"]
    m_verl_trainer_sft_trainer_ray["verl/trainer/sft_trainer_ray"]
    m_verl_utils___init__["verl/utils/__init__"]
    m_verl_utils_activation_offload["verl/utils/activation_offload"]
    m_verl_utils_attention_utils["verl/utils/attention_utils"]
    m_verl_utils_chat_template["verl/utils/chat_template"]
    m_verl_utils_checkpoint___init__["verl/utils/checkpoint/__init__"]
    m_verl_utils_checkpoint_checkpoint_handler["verl/utils/checkpoint/checkpoint_handler"]
    m_verl_utils_checkpoint_checkpoint_manager["verl/utils/checkpoint/checkpoint_manager"]
    m_verl_utils_checkpoint_fsdp_checkpoint_manager["verl/utils/checkpoint/fsdp_checkpoint_manager"]
    m_verl_utils_checkpoint_megatron_checkpoint_manager["verl/utils/checkpoint/megatron_checkpoint_manager"]
    m_verl_utils_config["verl/utils/config"]
    m_verl_utils_dataset___init__["verl/utils/dataset/__init__"]
    m_verl_utils_dataset_dataset_utils["verl/utils/dataset/dataset_utils"]
    m_verl_utils_dataset_multiturn_sft_dataset["verl/utils/dataset/multiturn_sft_dataset"]
    m_verl_utils_dataset_rl_dataset["verl/utils/dataset/rl_dataset"]
    m_verl_utils_dataset_rm_dataset["verl/utils/dataset/rm_dataset"]
    m_verl_utils_dataset_vision_utils["verl/utils/dataset/vision_utils"]
    m_verl_utils_debug___init__["verl/utils/debug/__init__"]
    m_verl_utils_debug_metrics["verl/utils/debug/metrics"]
    m_verl_utils_debug_performance["verl/utils/debug/performance"]
    m_verl_utils_debug_trajectory_tracker["verl/utils/debug/trajectory_tracker"]
    m_verl_utils_device["verl/utils/device"]
    m_verl_utils_distributed["verl/utils/distributed"]
    m_verl_utils_experimental___init__["verl/utils/experimental/__init__"]
    m_verl_utils_experimental_reward_utils["verl/utils/experimental/reward_utils"]
    m_verl_utils_experimental_torch_functional["verl/utils/experimental/torch_functional"]
    m_verl_utils_flops_counter["verl/utils/flops_counter"]
    m_verl_utils_fp8_utils["verl/utils/fp8_utils"]
    m_verl_utils_fs["verl/utils/fs"]
    m_verl_utils_fsdp_utils["verl/utils/fsdp_utils"]
    m_verl_utils_groupwise["verl/utils/groupwise"]
    m_verl_utils_hdfs_io["verl/utils/hdfs_io"]
    m_verl_utils_import_utils["verl/utils/import_utils"]
    m_verl_utils_kernel___init__["verl/utils/kernel/__init__"]
    m_verl_utils_kernel_fp8_kernel["verl/utils/kernel/fp8_kernel"]
    m_verl_utils_kernel_kernels["verl/utils/kernel/kernels"]
    m_verl_utils_kernel_linear_cross_entropy["verl/utils/kernel/linear_cross_entropy"]
    m_verl_utils_logger___init__["verl/utils/logger/__init__"]
    m_verl_utils_logger_aggregate_logger["verl/utils/logger/aggregate_logger"]
    m_verl_utils_logging_utils["verl/utils/logging_utils"]
    m_verl_utils_megatron___init__["verl/utils/megatron/__init__"]
    m_verl_utils_megatron_dist_checkpointing["verl/utils/megatron/dist_checkpointing"]
    m_verl_utils_megatron_memory["verl/utils/megatron/memory"]
    m_verl_utils_megatron_optimizer["verl/utils/megatron/optimizer"]
    m_verl_utils_megatron_pipeline_parallel["verl/utils/megatron/pipeline_parallel"]
    m_verl_utils_megatron_router_replay_patch["verl/utils/megatron/router_replay_patch"]
    m_verl_utils_megatron_router_replay_utils["verl/utils/megatron/router_replay_utils"]
    m_verl_utils_megatron_sequence_parallel["verl/utils/megatron/sequence_parallel"]
    m_verl_utils_megatron_tensor_parallel["verl/utils/megatron/tensor_parallel"]
    m_verl_utils_megatron_peft_utils["verl/utils/megatron_peft_utils"]
    m_verl_utils_megatron_utils["verl/utils/megatron_utils"]
    m_verl_utils_memory_utils["verl/utils/memory_utils"]
    m_verl_utils_metric___init__["verl/utils/metric/__init__"]
    m_verl_utils_metric_utils["verl/utils/metric/utils"]
    m_verl_utils_model["verl/utils/model"]
    m_verl_utils_modelopt___init__["verl/utils/modelopt/__init__"]
    m_verl_utils_modelopt_megatron_qat_patch["verl/utils/modelopt/megatron_qat_patch"]
    m_verl_utils_modelopt_qat_utils["verl/utils/modelopt/qat_utils"]
    m_verl_utils_modelopt_qat_weight_exporter["verl/utils/modelopt/qat_weight_exporter"]
    m_verl_utils_modelopt_quantize["verl/utils/modelopt/quantize"]
    m_verl_utils_modelopt_vllm_modelopt_patch["verl/utils/modelopt/vllm_modelopt_patch"]
    m_verl_utils_net_utils["verl/utils/net_utils"]
    m_verl_utils_npu_flash_attn_utils["verl/utils/npu_flash_attn_utils"]
    m_verl_utils_profiler___init__["verl/utils/profiler/__init__"]
    m_verl_utils_profiler_config["verl/utils/profiler/config"]
    m_verl_utils_profiler_empty_annotations["verl/utils/profiler/empty_annotations"]
    m_verl_utils_profiler_mstx_profile["verl/utils/profiler/mstx_profile"]
    m_verl_utils_profiler_nvtx_profile["verl/utils/profiler/nvtx_profile"]
    m_verl_utils_profiler_performance["verl/utils/profiler/performance"]
    m_verl_utils_profiler_profile["verl/utils/profiler/profile"]
    m_verl_utils_profiler_torch_profile["verl/utils/profiler/torch_profile"]
    m_verl_utils_py_functional["verl/utils/py_functional"]
    m_verl_utils_qat___init__["verl/utils/qat/__init__"]
    m_verl_utils_qat_core["verl/utils/qat/core"]
    m_verl_utils_qat_linear["verl/utils/qat/linear"]
    m_verl_utils_qat_quantizer["verl/utils/qat/quantizer"]
    m_verl_utils_qat_vllm_patch["verl/utils/qat/vllm_patch"]
    m_verl_utils_ray_utils["verl/utils/ray_utils"]
    m_verl_utils_rendezvous___init__["verl/utils/rendezvous/__init__"]
    m_verl_utils_rendezvous_ray_backend["verl/utils/rendezvous/ray_backend"]
    m_verl_utils_reward_score___init__["verl/utils/reward_score/__init__"]
    m_verl_utils_reward_score_geo3k["verl/utils/reward_score/geo3k"]
    m_verl_utils_reward_score_gsm8k["verl/utils/reward_score/gsm8k"]
    m_verl_utils_reward_score_jpeg_compressibility["verl/utils/reward_score/jpeg_compressibility"]
    m_verl_utils_reward_score_math_batch["verl/utils/reward_score/math_batch"]
    m_verl_utils_reward_score_math_dapo["verl/utils/reward_score/math_dapo"]
    m_verl_utils_reward_score_math_reward["verl/utils/reward_score/math_reward"]
    m_verl_utils_reward_score_math_verify["verl/utils/reward_score/math_verify"]
    m_verl_utils_reward_score_prime_code___init__["verl/utils/reward_score/prime_code/__init__"]
    m_verl_utils_reward_score_prime_code_testing_util["verl/utils/reward_score/prime_code/testing_util"]
    m_verl_utils_reward_score_prime_code_utils["verl/utils/reward_score/prime_code/utils"]
    m_verl_utils_reward_score_prime_math___init__["verl/utils/reward_score/prime_math/__init__"]
    m_verl_utils_reward_score_prime_math_grader["verl/utils/reward_score/prime_math/grader"]
    m_verl_utils_reward_score_prime_math_math_normalize["verl/utils/reward_score/prime_math/math_normalize"]
    m_verl_utils_reward_score_rlla["verl/utils/reward_score/rlla"]
    m_verl_utils_reward_score_sandbox_fusion___init__["verl/utils/reward_score/sandbox_fusion/__init__"]
    m_verl_utils_reward_score_sandbox_fusion_utils["verl/utils/reward_score/sandbox_fusion/utils"]
    m_verl_utils_reward_score_search_r1_like_qa_em["verl/utils/reward_score/search_r1_like_qa_em"]
    m_verl_utils_rollout_skip["verl/utils/rollout_skip"]
    m_verl_utils_rollout_trace["verl/utils/rollout_trace"]
    m_verl_utils_seqlen_balancing["verl/utils/seqlen_balancing"]
    m_verl_utils_sglang_sglang_fp8_utils["verl/utils/sglang/sglang_fp8_utils"]
    m_verl_utils_tensordict_utils["verl/utils/tensordict_utils"]
    m_verl_utils_tokenizer["verl/utils/tokenizer"]
    m_verl_utils_torch_dtypes["verl/utils/torch_dtypes"]
    m_verl_utils_torch_functional["verl/utils/torch_functional"]
    m_verl_utils_tracking["verl/utils/tracking"]
    m_verl_utils_transformers_compat["verl/utils/transformers_compat"]
    m_verl_utils_trtllm_trtllm_fp8_utils["verl/utils/trtllm/trtllm_fp8_utils"]
    m_verl_utils_ulysses["verl/utils/ulysses"]
    m_verl_utils_vllm___init__["verl/utils/vllm/__init__"]
    m_verl_utils_vllm_npu_vllm_patch["verl/utils/vllm/npu_vllm_patch"]
    m_verl_utils_vllm_patch["verl/utils/vllm/patch"]
    m_verl_utils_vllm_utils["verl/utils/vllm/utils"]
    m_verl_utils_vllm_vllm_fp8_utils["verl/utils/vllm/vllm_fp8_utils"]
    m_verl_utils_vllm_omni___init__["verl/utils/vllm_omni/__init__"]
    m_verl_utils_vllm_omni_utils["verl/utils/vllm_omni/utils"]
    m_verl_workers___init__["verl/workers/__init__"]
    m_verl_workers_actor___init__["verl/workers/actor/__init__"]
    m_verl_workers_actor_base["verl/workers/actor/base"]
    m_verl_workers_actor_dp_actor["verl/workers/actor/dp_actor"]
    m_verl_workers_actor_megatron_actor["verl/workers/actor/megatron_actor"]
    m_verl_workers_config___init__["verl/workers/config/__init__"]
    m_verl_workers_config_actor["verl/workers/config/actor"]
    m_verl_workers_config_critic["verl/workers/config/critic"]
    m_verl_workers_config_distillation["verl/workers/config/distillation"]
    m_verl_workers_config_engine["verl/workers/config/engine"]
    m_verl_workers_config_megatron_peft["verl/workers/config/megatron_peft"]
    m_verl_workers_config_model["verl/workers/config/model"]
    m_verl_workers_config_optimizer["verl/workers/config/optimizer"]
    m_verl_workers_config_reward["verl/workers/config/reward"]
    m_verl_workers_config_rollout["verl/workers/config/rollout"]
    m_verl_workers_critic___init__["verl/workers/critic/__init__"]
    m_verl_workers_critic_base["verl/workers/critic/base"]
    m_verl_workers_critic_dp_critic["verl/workers/critic/dp_critic"]
    m_verl_workers_critic_megatron_critic["verl/workers/critic/megatron_critic"]
    m_verl_workers_engine___init__["verl/workers/engine/__init__"]
    m_verl_workers_engine_automodel___init__["verl/workers/engine/automodel/__init__"]
    m_verl_workers_engine_automodel_transformer_impl["verl/workers/engine/automodel/transformer_impl"]
    m_verl_workers_engine_automodel_utils["verl/workers/engine/automodel/utils"]
    m_verl_workers_engine_base["verl/workers/engine/base"]
    m_verl_workers_engine_fsdp___init__["verl/workers/engine/fsdp/__init__"]
    m_verl_workers_engine_fsdp_transformer_impl["verl/workers/engine/fsdp/transformer_impl"]
    m_verl_workers_engine_fsdp_utils["verl/workers/engine/fsdp/utils"]
    m_verl_workers_engine_megatron___init__["verl/workers/engine/megatron/__init__"]
    m_verl_workers_engine_megatron_transformer_impl["verl/workers/engine/megatron/transformer_impl"]
    m_verl_workers_engine_megatron_utils["verl/workers/engine/megatron/utils"]
    m_verl_workers_engine_mindspeed___init__["verl/workers/engine/mindspeed/__init__"]
    m_verl_workers_engine_mindspeed_transformer_impl["verl/workers/engine/mindspeed/transformer_impl"]
    m_verl_workers_engine_torchtitan___init__["verl/workers/engine/torchtitan/__init__"]
    m_verl_workers_engine_torchtitan_transformer_impl["verl/workers/engine/torchtitan/transformer_impl"]
    m_verl_workers_engine_torchtitan_utils["verl/workers/engine/torchtitan/utils"]
    m_verl_workers_engine_utils["verl/workers/engine/utils"]
    m_verl_workers_engine_veomni___init__["verl/workers/engine/veomni/__init__"]
    m_verl_workers_engine_veomni_transformer_impl["verl/workers/engine/veomni/transformer_impl"]
    m_verl_workers_engine_veomni_utils["verl/workers/engine/veomni/utils"]
    m_verl_workers_engine_workers["verl/workers/engine_workers"]
    m_verl_workers_fsdp_workers["verl/workers/fsdp_workers"]
    m_verl_workers_megatron_workers["verl/workers/megatron_workers"]
    m_verl_workers_reward_manager___init__["verl/workers/reward_manager/__init__"]
    m_verl_workers_reward_manager_abstract["verl/workers/reward_manager/abstract"]
    m_verl_workers_reward_manager_batch["verl/workers/reward_manager/batch"]
    m_verl_workers_reward_manager_dapo["verl/workers/reward_manager/dapo"]
    m_verl_workers_reward_manager_naive["verl/workers/reward_manager/naive"]
    m_verl_workers_reward_manager_prime["verl/workers/reward_manager/prime"]
    m_verl_workers_reward_manager_registry["verl/workers/reward_manager/registry"]
    m_verl_workers_rollout___init__["verl/workers/rollout/__init__"]
    m_verl_workers_rollout_base["verl/workers/rollout/base"]
    m_verl_workers_rollout_hf_rollout["verl/workers/rollout/hf_rollout"]
    m_verl_workers_rollout_naive___init__["verl/workers/rollout/naive/__init__"]
    m_verl_workers_rollout_naive_naive_rollout["verl/workers/rollout/naive/naive_rollout"]
    m_verl_workers_rollout_replica["verl/workers/rollout/replica"]
    m_verl_workers_rollout_schemas["verl/workers/rollout/schemas"]
    m_verl_workers_rollout_sglang_rollout___init__["verl/workers/rollout/sglang_rollout/__init__"]
    m_verl_workers_rollout_sglang_rollout_async_sglang_server["verl/workers/rollout/sglang_rollout/async_sglang_server"]
    m_verl_workers_rollout_sglang_rollout_http_server_engine["verl/workers/rollout/sglang_rollout/http_server_engine"]
    m_verl_workers_rollout_sglang_rollout_sglang_rollout["verl/workers/rollout/sglang_rollout/sglang_rollout"]
    m_verl_workers_rollout_sglang_rollout_utils["verl/workers/rollout/sglang_rollout/utils"]
    m_verl_workers_rollout_tokenizer["verl/workers/rollout/tokenizer"]
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server["verl/workers/rollout/trtllm_rollout/trtllm_async_server"]
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout["verl/workers/rollout/trtllm_rollout/trtllm_rollout"]
    m_verl_workers_rollout_trtllm_rollout_trtllm_worker_extension["verl/workers/rollout/trtllm_rollout/trtllm_worker_extension"]
    m_verl_workers_rollout_utils["verl/workers/rollout/utils"]
    m_verl_workers_rollout_vllm_rollout___init__["verl/workers/rollout/vllm_rollout/__init__"]
    m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer["verl/workers/rollout/vllm_rollout/bucketed_weight_transfer"]
    m_verl_workers_rollout_vllm_rollout_utils["verl/workers/rollout/vllm_rollout/utils"]
    m_verl_workers_rollout_vllm_rollout_vllm_async_server["verl/workers/rollout/vllm_rollout/vllm_async_server"]
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server["verl/workers/rollout/vllm_rollout/vllm_omni_async_server"]
    m_verl_workers_rollout_vllm_rollout_vllm_rollout["verl/workers/rollout/vllm_rollout/vllm_rollout"]
    m_verl_workers_sharding_manager___init__["verl/workers/sharding_manager/__init__"]
    m_verl_workers_sharding_manager_base["verl/workers/sharding_manager/base"]
    m_verl_workers_sharding_manager_fsdp_ulysses["verl/workers/sharding_manager/fsdp_ulysses"]
    m_verl_workers_utils___init__["verl/workers/utils/__init__"]
    m_verl_workers_utils_losses["verl/workers/utils/losses"]
    m_verl_workers_utils_padding["verl/workers/utils/padding"]
    m_examples_data_preprocess_aime2024_multiturn_w_tool --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_dapo_multiturn_w_tool --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_full_hh_rlhf --> m_verl_utils_fs
    m_examples_data_preprocess_geo3k --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_geo3k_multiturn_w_tool --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_gsm8k --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_gsm8k_multiturn_sft --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_gsm8k_multiturn_w_interaction --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_gsm8k_multiturn_w_tool --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_gsm8k_tool_agent_loop --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_hellaswag --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_math_dataset --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_math_dataset --> m_verl_utils_reward_score_math_reward
    m_examples_data_preprocess_multiturn --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_pokemon --> m_verl_utils_hdfs_io
    m_examples_data_preprocess_preprocess_search_r1_dataset --> m_my_recipes_rllm_utils
    m_examples_data_preprocess_preprocess_search_r1_dataset --> m_verl_utils_hdfs_io
    m_examples_fapo_trainer_prepare_data --> m_verl_utils_hdfs_io
    m_examples_fapo_trainer_reward_fn --> m_verl_utils_ray_utils
    m_examples_fapo_trainer_reward_fn --> m_verl_utils_reward_score_math_dapo
    m_examples_flowgrpo_trainer_reward_fn --> m_verl_utils_experimental_reward_utils
    m_examples_flowgrpo_trainer_reward_fn --> m_verl_utils_ray_utils
    m_examples_sglang_multiturn_gsm8k_toolcall_shaping_gsm8k_toolcall_shaping --> m_verl_utils_reward_score_gsm8k
    m_examples_split_placement_main_ppo_split --> m_examples_split_placement_split_monkey_patch
    m_examples_split_placement_main_ppo_split --> m_my_recipes_rllm_utils
    m_examples_split_placement_main_ppo_split --> m_verl_trainer_ppo_ray_trainer
    m_examples_split_placement_main_ppo_split --> m_verl_trainer_ppo_utils
    m_examples_split_placement_main_ppo_split --> m_verl_utils_fs
    m_examples_split_placement_main_ppo_split --> m_verl_workers_fsdp_workers
    m_examples_split_placement_main_ppo_split --> m_verl_workers_megatron_workers
    m_examples_split_placement_split_monkey_patch --> m_verl_trainer_ppo_ray_trainer
    m_examples_split_placement_split_monkey_patch --> m_verl_trainer_ppo_reward
    m_examples_split_placement_split_monkey_patch --> m_verl_utils_tracking
    m_examples_tutorial_agent_loop_get_started_sandbox --> m_my_recipes_rllm_utils
    m_examples_tutorial_agent_loop_get_started_sandbox --> m_verl_tools_base_tool
    m_examples_vllm_omni_pipeline_qwenimage --> m_examples_vllm_omni_scheduling_flow_match_sde_discrete
    m_examples_vllm_omni_pipeline_qwenimage --> m_my_recipes_rllm_utils
    m_examples_vllm_omni_scheduling_flow_match_sde_discrete --> m_my_recipes_rllm_utils
    m_my_recipes_genrm_remote_reward_function --> m_verl_utils_reward_score_math_reward
    m_my_recipes_genrm_remote_reward_function_genrm --> m_verl_utils_reward_score_math_reward
    m_my_recipes_insturct_following_evaluation_main --> m_my_recipes_insturct_following_instructions_registry
    m_my_recipes_insturct_following_instructions --> m_my_recipes_insturct_following_instructions_util
    m_my_recipes_insturct_following_instructions_registry --> m_my_recipes_insturct_following_instructions
    m_my_recipes_insturct_following_reward_function --> m_my_recipes_insturct_following_instructions_registry
    m_my_recipes_rllm_rewards___init__ --> m_my_recipes_rllm_rewards_reward_types
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_code_utils_codeforces
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_code_utils_firejail_exec
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_code_utils_humanevalplus
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_code_utils_kodcode
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_code_utils_livecodebench
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_code_utils_taco
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_rewards_reward_types
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_tools_code_tools_code_tool
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_tools_code_tools_together_tool
    m_my_recipes_rllm_rewards_code_reward --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_code_utils_codeforces --> m_my_recipes_rllm_rewards_code_utils_pyext2
    m_my_recipes_rllm_rewards_code_utils_firejail_exec --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_code_utils_humanevalplus --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_code_utils_kodcode --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_code_utils_livecodebench --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_code_utils_taco --> m_my_recipes_rllm_rewards_code_utils_pyext2
    m_my_recipes_rllm_rewards_code_utils_taco --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_math_reward --> m_my_recipes_rllm_globals
    m_my_recipes_rllm_rewards_math_reward --> m_my_recipes_rllm_system_prompts
    m_my_recipes_rllm_rewards_math_reward --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_math_utils___init__ --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_rewards_rl_reward --> m_my_recipes_rllm_rewards_code_reward
    m_my_recipes_rllm_rewards_rl_reward --> m_my_recipes_rllm_rewards_math_reward
    m_my_recipes_rllm_rewards_rl_reward --> m_my_recipes_rllm_rewards_reward_types
    m_my_recipes_rllm_tools_code_tools___init__ --> m_my_recipes_rllm_tools_code_tools_e2b_tool
    m_my_recipes_rllm_tools_code_tools___init__ --> m_my_recipes_rllm_tools_code_tools_lcb_tool
    m_my_recipes_rllm_tools_code_tools___init__ --> m_my_recipes_rllm_tools_code_tools_local_tool
    m_my_recipes_rllm_tools_code_tools_code_tool --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_tools_code_tools_e2b_tool --> m_my_recipes_rllm_tools_code_tools_code_tool
    m_my_recipes_rllm_tools_code_tools_lcb_tool --> m_my_recipes_rllm_rewards_code_utils_livecodebench
    m_my_recipes_rllm_tools_code_tools_lcb_tool --> m_my_recipes_rllm_tools_code_tools_code_tool
    m_my_recipes_rllm_tools_code_tools_local_tool --> m_my_recipes_rllm_tools_code_tools_code_tool
    m_my_recipes_rllm_tools_code_tools_together_tool --> m_my_recipes_rllm_tools_code_tools_code_tool
    m_my_recipes_rllm_tools_code_tools_utils --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_tools_example_tool --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_tools_math_tools___init__ --> m_my_recipes_rllm_tools_math_tools_calculator
    m_my_recipes_rllm_tools_math_tools_calculator --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_tools_multi_tool --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_tools_tool_base --> m_my_recipes_rllm_utils
    m_my_recipes_rllm_tools_web_tools___init__ --> m_my_recipes_rllm_tools_web_tools_firecrawl_tool
    m_my_recipes_rllm_tools_web_tools___init__ --> m_my_recipes_rllm_tools_web_tools_gsearch_tool
    m_my_recipes_rllm_tools_web_tools___init__ --> m_my_recipes_rllm_tools_web_tools_tavily_tool
    m_my_recipes_rllm_tools_web_tools_firecrawl_tool --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_tools_web_tools_gsearch_tool --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_tools_web_tools_tavily_tool --> m_my_recipes_rllm_tools_tool_base
    m_my_recipes_rllm_utils --> m_my_recipes_rllm_globals
    m_my_recipes_syntool___init__ --> m_my_recipes_syntool_agent_loop
    m_my_recipes_syntool___init__ --> m_my_recipes_syntool_dataset
    m_my_recipes_syntool___init__ --> m_my_recipes_syntool_reward
    m_my_recipes_syntool___init__ --> m_my_recipes_syntool_tool
    m_my_recipes_syntool_agent_loop --> m_my_recipes_syntool_tool
    m_my_recipes_syntool_agent_loop --> m_verl_experimental_agent_loop_agent_loop
    m_my_recipes_syntool_agent_loop --> m_verl_experimental_agent_loop_tool_agent_loop
    m_my_recipes_syntool_agent_loop --> m_verl_tools_schemas
    m_my_recipes_syntool_dataset --> m_verl_utils_dataset_rl_dataset
    m_my_recipes_syntool_dataset --> m_verl_utils_tokenizer
    m_my_recipes_syntool_tool --> m_verl_tools_base_tool
    m_my_recipes_syntool_tool --> m_verl_tools_schemas
    m_my_recipes_syntool_tool --> m_verl_utils_rollout_trace
    m_scripts_converter_hf_to_mcore --> m_verl_model_merger_megatron_model_merger
    m_scripts_converter_hf_to_mcore --> m_verl_models_mcore_loader
    m_scripts_converter_hf_to_mcore --> m_verl_utils_device
    m_scripts_converter_hf_to_mcore --> m_verl_utils_distributed
    m_scripts_converter_hf_to_mcore --> m_verl_utils_megatron_utils
    m_scripts_converter_hf_to_mcore --> m_verl_utils_qat_core
    m_scripts_legacy_model_merger --> m_my_recipes_rllm_utils
    m_scripts_legacy_model_merger --> m_verl_utils_megatron_utils
    m_scripts_legacy_model_merger --> m_verl_utils_transformers_compat
    m_scripts_megatron_merge_lora --> m_verl_single_controller_base_decorator
    m_scripts_megatron_merge_lora --> m_verl_utils_megatron_utils
    m_scripts_megatron_merge_lora --> m_verl_workers_megatron_workers
    m_scripts_print_cfg --> m_verl_utils_config
    m_tests_checkpoint_engine_test_correctness_on_gpu --> m_tests_checkpoint_engine_test_utils
    m_tests_checkpoint_engine_test_correctness_on_gpu --> m_verl_single_controller_ray_base
    m_tests_checkpoint_engine_test_correctness_on_gpu --> m_verl_trainer_config_config
    m_tests_checkpoint_engine_test_correctness_on_gpu --> m_verl_utils_device
    m_tests_checkpoint_engine_test_correctness_on_gpu --> m_verl_utils_ray_utils
    m_tests_checkpoint_engine_test_correctness_on_npu --> m_tests_checkpoint_engine_test_utils
    m_tests_checkpoint_engine_test_correctness_on_npu --> m_verl_single_controller_ray_base
    m_tests_checkpoint_engine_test_correctness_on_npu --> m_verl_trainer_config_config
    m_tests_checkpoint_engine_test_correctness_on_npu --> m_verl_utils_device
    m_tests_checkpoint_engine_test_correctness_on_npu --> m_verl_utils_ray_utils
    m_tests_checkpoint_engine_test_special_server_adapter --> m_tests_checkpoint_engine_test_utils
    m_tests_checkpoint_engine_test_special_server_adapter --> m_verl_experimental_agent_loop_agent_loop
    m_tests_checkpoint_engine_test_special_server_adapter --> m_verl_experimental_fully_async_policy_agent_loop_agent_loop
    m_tests_checkpoint_engine_test_special_server_adapter --> m_verl_trainer_config_config
    m_tests_checkpoint_engine_test_special_server_adapter --> m_verl_utils_config
    m_tests_checkpoint_engine_test_utils --> m_verl_single_controller_base_decorator
    m_tests_checkpoint_engine_test_utils --> m_verl_trainer_config_config
    m_tests_checkpoint_engine_test_utils --> m_verl_utils_device
    m_tests_checkpoint_engine_test_utils --> m_verl_utils_fs
    m_tests_checkpoint_engine_test_utils --> m_verl_workers_config_rollout
    m_tests_checkpoint_engine_test_utils --> m_verl_workers_engine_workers
    m_tests_experimental_agent_loop_agent_utils --> m_my_recipes_rllm_utils
    m_tests_experimental_agent_loop_agent_utils --> m_my_recipes_syntool_agent_loop
    m_tests_experimental_agent_loop_agent_utils --> m_verl_experimental_reward_loop_reward_loop
    m_tests_experimental_agent_loop_agent_utils --> m_verl_single_controller_ray_base
    m_tests_experimental_agent_loop_agent_utils --> m_verl_trainer_ppo_ray_trainer
    m_tests_experimental_agent_loop_agent_utils --> m_verl_workers_fsdp_workers
    m_tests_experimental_agent_loop_test_agent_loop_extra_fields_schema_on_cpu --> m_verl_experimental_agent_loop_agent_loop
    m_tests_experimental_agent_loop_test_agent_loop_extra_fields_schema_on_cpu --> m_verl_experimental_agent_loop_single_turn_agent_loop
    m_tests_experimental_agent_loop_test_agent_loop_extra_fields_schema_on_cpu --> m_verl_utils_dataset_rl_dataset
    m_tests_experimental_agent_loop_test_agent_loop_extra_fields_schema_on_cpu --> m_verl_workers_rollout_replica
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_my_recipes_rllm_utils
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_tests_experimental_agent_loop_agent_utils
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_verl_experimental_agent_loop_agent_loop
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_verl_protocol
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_verl_tools_base_tool
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_verl_tools_schemas
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_verl_trainer_config_config
    m_tests_experimental_agent_loop_test_basic_agent_loop --> m_verl_utils_config
    m_tests_experimental_agent_loop_test_gpt_oss_tool_parser --> m_verl_experimental_agent_loop_tool_parser
    m_tests_experimental_agent_loop_test_multi_modal --> m_my_recipes_rllm_utils
    m_tests_experimental_agent_loop_test_multi_modal --> m_tests_experimental_agent_loop_agent_utils
    m_tests_experimental_agent_loop_test_multi_modal --> m_verl_protocol
    m_tests_experimental_agent_loop_test_multi_modal --> m_verl_tools_base_tool
    m_tests_experimental_agent_loop_test_multi_modal --> m_verl_tools_schemas
    m_tests_experimental_agent_loop_test_standalone_rollout --> m_my_recipes_rllm_utils
    m_tests_experimental_agent_loop_test_standalone_rollout --> m_tests_experimental_agent_loop_agent_utils
    m_tests_experimental_agent_loop_test_standalone_rollout --> m_verl_workers_rollout_replica
    m_tests_experimental_reward_loop_reward_fn --> m_verl_utils_reward_score_math_verify
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_my_recipes_rllm_utils
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_my_recipes_syntool_agent_loop
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_experimental_reward_loop_reward_loop
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_protocol
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_trainer_main_ppo
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_trainer_ppo_ray_trainer
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_utils_dataset_rl_dataset
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_utils_device
    m_tests_experimental_reward_loop_test_agent_reward_loop_colocate --> m_verl_workers_fsdp_workers
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone --> m_my_recipes_rllm_utils
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone --> m_my_recipes_syntool_agent_loop
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone --> m_verl_experimental_reward_loop_reward_loop
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone --> m_verl_protocol
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone --> m_verl_trainer_main_ppo
    m_tests_experimental_reward_loop_test_agent_reward_loop_standalone --> m_verl_utils_dataset_rl_dataset
    m_tests_experimental_reward_loop_test_async_token_bucket_on_cpu --> m_verl_experimental_reward_loop_reward_manager_limited
    m_tests_experimental_reward_loop_test_math_verify --> m_tests_experimental_agent_loop_agent_utils
    m_tests_experimental_reward_loop_test_math_verify --> m_verl_protocol
    m_tests_experimental_reward_loop_test_math_verify --> m_verl_trainer_main_ppo
    m_tests_experimental_reward_loop_test_math_verify --> m_verl_utils_dataset_rl_dataset
    m_tests_experimental_reward_loop_test_rate_limited_reward_manager_on_cpu --> m_verl_experimental_reward_loop_reward_manager_limited
    m_tests_experimental_reward_loop_test_reward_model_disrm --> m_my_recipes_rllm_utils
    m_tests_experimental_reward_loop_test_reward_model_disrm --> m_verl_experimental_reward_loop_reward_loop
    m_tests_experimental_reward_loop_test_reward_model_disrm --> m_verl_protocol
    m_tests_experimental_reward_loop_test_reward_model_disrm --> m_verl_utils_model
    m_tests_experimental_reward_loop_test_reward_model_disrm --> m_verl_utils_tokenizer
    m_tests_experimental_reward_loop_test_reward_model_genrm --> m_my_recipes_rllm_utils
    m_tests_experimental_reward_loop_test_reward_model_genrm --> m_verl_experimental_reward_loop_reward_loop
    m_tests_experimental_reward_loop_test_reward_model_genrm --> m_verl_protocol
    m_tests_experimental_reward_loop_test_reward_model_genrm --> m_verl_utils_model
    m_tests_experimental_reward_loop_test_reward_model_genrm --> m_verl_utils_tokenizer
    m_tests_experimental_reward_loop_test_visual_reward_manager --> m_my_recipes_rllm_utils
    m_tests_experimental_reward_loop_test_visual_reward_manager --> m_verl_experimental_reward_loop_reward_loop
    m_tests_experimental_reward_loop_test_visual_reward_manager --> m_verl_protocol
    m_tests_experimental_vla_test_sim_envs --> m_verl_experimental_vla_envs_isaac_env_isaac_env
    m_tests_experimental_vla_test_sim_envs --> m_verl_experimental_vla_envs_libero_env_libero_env
    m_tests_interactions_test_gsm8k_interaction --> m_verl_interactions_base
    m_tests_interactions_test_gsm8k_interaction --> m_verl_interactions_gsm8k_interaction
    m_tests_interactions_test_interaction_registry --> m_verl_interactions_base
    m_tests_interactions_test_interaction_registry --> m_verl_interactions_gsm8k_interaction
    m_tests_interactions_test_interaction_registry --> m_verl_interactions_utils_interaction_registry
    m_tests_models_test_engine --> m_my_recipes_rllm_utils
    m_tests_models_test_engine --> m_verl_trainer_config_config
    m_tests_models_test_engine --> m_verl_utils_distributed
    m_tests_models_test_engine --> m_verl_utils_model
    m_tests_models_test_engine --> m_verl_utils_torch_functional
    m_tests_models_test_engine --> m_verl_workers_config_engine
    m_tests_models_test_engine --> m_verl_workers_engine_workers
    m_tests_models_test_engine --> m_verl_workers_utils_losses
    m_tests_models_test_engine --> m_verl_workers_utils_padding
    m_tests_models_test_liger_vl_compat --> m_verl_models_transformers_monkey_patch
    m_tests_models_test_tiled_mlp_accuracy --> m_verl_models_transformers_tiled_mlp
    m_tests_models_test_tiled_mlp_accuracy --> m_verl_utils_distributed
    m_tests_models_test_transformer --> m_verl_utils_attention_utils
    m_tests_models_test_transformer --> m_verl_utils_device
    m_tests_models_test_transformer --> m_verl_utils_model
    m_tests_models_test_transformer --> m_verl_utils_torch_functional
    m_tests_models_test_transformers_ulysses --> m_verl_models_transformers_monkey_patch
    m_tests_models_test_transformers_ulysses --> m_verl_protocol
    m_tests_models_test_transformers_ulysses --> m_verl_utils_attention_utils
    m_tests_models_test_transformers_ulysses --> m_verl_utils_device
    m_tests_models_test_transformers_ulysses --> m_verl_utils_distributed
    m_tests_models_test_transformers_ulysses --> m_verl_utils_model
    m_tests_models_test_transformers_ulysses --> m_verl_utils_ulysses
    m_tests_models_test_transformers_ulysses --> m_verl_workers_sharding_manager_fsdp_ulysses
    m_tests_my_recipes_test_syntool_recipe --> m_my_recipes_syntool_agent_loop
    m_tests_my_recipes_test_syntool_recipe --> m_my_recipes_syntool_dataset
    m_tests_my_recipes_test_syntool_recipe --> m_my_recipes_syntool_reward
    m_tests_my_recipes_test_syntool_recipe --> m_my_recipes_syntool_tool
    m_tests_my_recipes_test_syntool_recipe --> m_verl_experimental_agent_loop_agent_loop
    m_tests_my_recipes_test_syntool_recipe --> m_verl_experimental_agent_loop_tool_parser
    m_tests_my_recipes_test_syntool_recipe --> m_verl_tools_schemas
    m_tests_my_recipes_test_syntool_recipe --> m_verl_workers_rollout_replica
    m_tests_single_controller_base_test_decorator --> m_verl_single_controller_base_decorator
    m_tests_single_controller_check_worker_alive_main --> m_verl_single_controller_base_decorator
    m_tests_single_controller_check_worker_alive_main --> m_verl_single_controller_base_worker
    m_tests_single_controller_check_worker_alive_main --> m_verl_single_controller_ray_base
    m_tests_single_controller_detached_worker_client --> m_tests_single_controller_detached_worker_server
    m_tests_single_controller_detached_worker_server --> m_verl_checkpoint_engine_base
    m_tests_single_controller_detached_worker_server --> m_verl_single_controller_base_decorator
    m_tests_single_controller_detached_worker_server --> m_verl_utils_megatron_optimizer
    m_tests_single_controller_detached_worker_server --> m_verl_utils_megatron_utils
    m_tests_single_controller_detached_worker_server --> m_verl_utils_qat_core
    m_tests_single_controller_test_auto_padding_on_cpu --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_auto_padding_on_cpu --> m_verl_protocol
    m_tests_single_controller_test_auto_padding_on_cpu --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_auto_padding_on_cpu --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_colocated_workers --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_colocated_workers --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_colocated_workers --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_colocated_workers --> m_verl_utils_device
    m_tests_single_controller_test_colocated_workers_fused --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_colocated_workers_fused --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_colocated_workers_fused --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_colocated_workers_fused --> m_verl_utils_device
    m_tests_single_controller_test_data_transfer --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_data_transfer --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_data_transfer --> m_verl_utils_device
    m_tests_single_controller_test_data_transfer --> m_verl_utils_ray_utils
    m_tests_single_controller_test_decorator_on_cpu --> m_my_recipes_rllm_utils
    m_tests_single_controller_test_decorator_on_cpu --> m_verl_protocol
    m_tests_single_controller_test_decorator_on_cpu --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_decorator_on_cpu --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_device_mesh_register --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_device_mesh_register --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_device_mesh_register --> m_verl_utils_device
    m_tests_single_controller_test_device_mesh_register --> m_verl_utils_distributed
    m_tests_single_controller_test_device_mesh_register --> m_verl_utils_tensordict_utils
    m_tests_single_controller_test_driverfunc_to_worker --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_driverfunc_to_worker --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_driverfunc_to_worker --> m_verl_utils_device
    m_tests_single_controller_test_fused_workers_on_cpu --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_fused_workers_on_cpu --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_fused_workers_on_cpu --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_get_set_dispatch_collect_cpu --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_high_level_scheduling_api --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_high_level_scheduling_api --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_high_level_scheduling_api --> m_verl_utils_device
    m_tests_single_controller_test_nested_worker --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_nested_worker --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_nested_worker --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_nested_worker --> m_verl_utils_device
    m_tests_single_controller_test_ray_collectives --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_ray_collectives --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_ray_local_envs_on_cpu --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_ray_local_envs_on_cpu --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_ray_utils_on_cpu --> m_verl_utils_ray_utils
    m_tests_single_controller_test_rvdz --> m_verl_utils_rendezvous_ray_backend
    m_tests_single_controller_test_split_resource_pool --> m_verl_checkpoint_engine_base
    m_tests_single_controller_test_split_resource_pool --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_split_resource_pool --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_split_resource_pool --> m_verl_utils_device
    m_tests_single_controller_test_worker_group_basics --> m_verl_single_controller_base_decorator
    m_tests_single_controller_test_worker_group_basics --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_worker_group_basics --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_worker_group_basics --> m_verl_utils_device
    m_tests_single_controller_test_worker_group_torch --> m_verl_single_controller_base_worker
    m_tests_single_controller_test_worker_group_torch --> m_verl_single_controller_ray_base
    m_tests_single_controller_test_worker_group_torch --> m_verl_utils_device
    m_tests_single_controller_test_worker_group_torch --> m_verl_utils_distributed
    m_tests_special_distributed_test_fsdp_ckpt --> m_verl_utils_attention_utils
    m_tests_special_distributed_test_fsdp_ckpt --> m_verl_utils_checkpoint_fsdp_checkpoint_manager
    m_tests_special_distributed_test_fsdp_ckpt --> m_verl_utils_device
    m_tests_special_distributed_test_fsdp_ckpt --> m_verl_utils_distributed
    m_tests_special_distributed_test_fsdp_ckpt --> m_verl_utils_fsdp_utils
    m_tests_special_distributed_test_fsdp_ckpt --> m_verl_utils_model
    m_tests_special_distributed_test_mcore_config_converter --> m_verl_utils_distributed
    m_tests_special_distributed_test_tensor_dict --> m_verl_protocol
    m_tests_special_distributed_test_tensor_dict --> m_verl_utils_device
    m_tests_special_distributed_test_tensor_dict --> m_verl_utils_distributed
    m_tests_special_distributed_test_tensor_dict --> m_verl_utils_megatron_tensor_parallel
    m_tests_special_distributed_test_tensor_dict --> m_verl_utils_qat_core
    m_tests_special_distributed_test_tensor_dict --> m_verl_utils_torch_functional
    m_tests_special_distributed_test_torch_functional --> m_verl_utils_torch_functional
    m_tests_special_e2e_envs_digit_completion___init__ --> m_tests_special_e2e_envs_digit_completion_task
    m_tests_special_e2e_envs_digit_completion___init__ --> m_tests_special_e2e_envs_digit_completion_tokenizer
    m_tests_test_base_config_on_cpu --> m_verl_base_config
    m_tests_test_protocol_on_cpu --> m_my_recipes_rllm_utils
    m_tests_test_protocol_on_cpu --> m_verl_protocol
    m_tests_test_protocol_v2_on_cpu --> m_my_recipes_rllm_utils
    m_tests_trainer_config_test_algo_config_on_cpu --> m_verl_trainer_config_config
    m_tests_trainer_config_test_algo_config_on_cpu --> m_verl_trainer_ppo_core_algos
    m_tests_trainer_config_test_algo_config_on_cpu --> m_verl_utils_config
    m_tests_trainer_ppo_test_core_algos_on_cpu --> m_verl_trainer_ppo_core_algos
    m_tests_trainer_ppo_test_metric_utils_on_cpu --> m_verl_trainer_ppo_metric_utils
    m_tests_trainer_ppo_test_metric_utils_on_cpu --> m_verl_utils_metric_utils
    m_tests_trainer_ppo_test_rollout_corr --> m_verl_trainer_ppo_rollout_corr_helper
    m_tests_trainer_ppo_test_rollout_corr_integration --> m_verl_protocol
    m_tests_trainer_ppo_test_rollout_corr_integration --> m_verl_trainer_config_algorithm
    m_tests_trainer_ppo_test_rollout_corr_integration --> m_verl_trainer_ppo_core_algos
    m_tests_trainer_ppo_test_rollout_corr_integration --> m_verl_trainer_ppo_rollout_corr_helper
    m_tests_trainer_ppo_test_rollout_corr_integration --> m_verl_workers_config_actor
    m_tests_utils_ckpt_test_checkpoint_cleanup_on_cpu --> m_verl_utils_checkpoint_checkpoint_manager
    m_tests_utils_ckpt_test_checkpoint_cleanup_on_cpu --> m_verl_utils_distributed
    m_tests_utils_ckpt_test_esi_save_ckpt_on_cpu --> m_verl_utils_checkpoint_checkpoint_manager
    m_tests_utils_dataset_test_create_rl_sampler_on_cpu --> m_verl_experimental_dataset_sampler
    m_tests_utils_dataset_test_create_rl_sampler_on_cpu --> m_verl_trainer_main_ppo
    m_tests_utils_dataset_test_multiturn_sft_dataset_on_cpu --> m_my_recipes_rllm_utils
    m_tests_utils_dataset_test_multiturn_sft_dataset_on_cpu --> m_verl_utils_dataset_dataset_utils
    m_tests_utils_dataset_test_multiturn_sft_dataset_on_cpu --> m_verl_utils_dataset_multiturn_sft_dataset
    m_tests_utils_dataset_test_multiturn_sft_dataset_on_cpu --> m_verl_utils_model
    m_tests_utils_dataset_test_rl_collate_fn_on_cpu --> m_verl_utils_dataset_rl_dataset
    m_tests_utils_dataset_test_rl_dataset_on_cpu --> m_my_recipes_rllm_utils
    m_tests_utils_dataset_test_rl_dataset_on_cpu --> m_verl_utils_dataset_rl_dataset
    m_tests_utils_debug_test_metrics --> m_verl_protocol
    m_tests_utils_debug_test_metrics --> m_verl_utils_debug_metrics
    m_tests_utils_megatron_test_pipeline_parallel --> m_verl_model_merger_megatron_model_merger
    m_tests_utils_megatron_test_pipeline_parallel --> m_verl_utils_megatron_pipeline_parallel
    m_tests_utils_reward_score_reward_score_test_sandbox_fusion_on_cpu --> m_verl_utils_reward_score_sandbox_fusion_utils
    m_tests_utils_reward_score_test_sandbox_on_cpu --> m_verl_workers_reward_manager_prime
    m_tests_utils_test_activation_offload --> m_verl_utils_activation_offload
    m_tests_utils_test_activation_offload --> m_verl_utils_attention_utils
    m_tests_utils_test_activation_offload --> m_verl_utils_checkpoint_fsdp_checkpoint_manager
    m_tests_utils_test_activation_offload --> m_verl_utils_device
    m_tests_utils_test_activation_offload --> m_verl_utils_distributed
    m_tests_utils_test_activation_offload --> m_verl_utils_fsdp_utils
    m_tests_utils_test_activation_offload --> m_verl_utils_model
    m_tests_utils_test_bucketed_weight_transfer --> m_verl_utils_device
    m_tests_utils_test_bucketed_weight_transfer --> m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer
    m_tests_utils_test_check_ipc_version_support_on_npu --> m_verl_utils_device
    m_tests_utils_test_config_on_cpu --> m_my_recipes_rllm_utils
    m_tests_utils_test_config_on_cpu --> m_verl_base_config
    m_tests_utils_test_flops_counter --> m_verl_utils_flops_counter
    m_tests_utils_test_fs_on_cpu --> m_verl_utils_fs
    m_tests_utils_test_fsdp2_peft_wrapping --> m_verl_utils_fsdp_utils
    m_tests_utils_test_fsdp_lora_merge --> m_verl_utils_device
    m_tests_utils_test_fsdp_lora_merge --> m_verl_utils_distributed
    m_tests_utils_test_fsdp_lora_merge --> m_verl_utils_fsdp_utils
    m_tests_utils_test_groupwise --> m_my_recipes_rllm_utils
    m_tests_utils_test_groupwise --> m_verl_utils_device
    m_tests_utils_test_import_utils_on_cpu --> m_verl_utils_import_utils
    m_tests_utils_test_linear_cross_entropy --> m_verl_utils_device
    m_tests_utils_test_linear_cross_entropy --> m_verl_utils_experimental_torch_functional
    m_tests_utils_test_linear_cross_entropy --> m_verl_utils_kernel_linear_cross_entropy
    m_tests_utils_test_linear_cross_entropy --> m_verl_utils_torch_functional
    m_tests_utils_test_mlflow_key_sanitization --> m_verl_utils_tracking
    m_tests_utils_test_model_on_cpu --> m_verl_utils_model
    m_tests_utils_test_normalize_peft_param_name --> m_verl_utils_device
    m_tests_utils_test_normalize_peft_param_name --> m_verl_utils_distributed
    m_tests_utils_test_normalize_peft_param_name --> m_verl_utils_fsdp_utils
    m_tests_utils_test_normalize_peft_param_name --> m_verl_utils_model
    m_tests_utils_test_normalize_peft_param_name_on_cpu --> m_verl_utils_fsdp_utils
    m_tests_utils_test_nvtx_profile --> m_my_recipes_rllm_utils
    m_tests_utils_test_nvtx_profile --> m_verl_utils_profiler_config
    m_tests_utils_test_nvtx_profile --> m_verl_utils_profiler_profile
    m_tests_utils_test_padding_on_cpu --> m_verl_workers_utils_padding
    m_tests_utils_test_prepare_micro_batches_with_group_size --> m_my_recipes_rllm_utils
    m_tests_utils_test_prepare_micro_batches_with_group_size --> m_verl_workers_engine_utils
    m_tests_utils_test_rollout_skip_on_cpu --> m_verl_protocol
    m_tests_utils_test_rollout_skip_on_cpu --> m_verl_utils_rollout_skip
    m_tests_utils_test_rollout_trace_on_cpu --> m_verl_utils_rollout_trace
    m_tests_utils_test_seqlen_balancing --> m_verl_utils_device
    m_tests_utils_test_seqlen_balancing --> m_verl_utils_distributed
    m_tests_utils_test_seqlen_balancing --> m_verl_utils_model
    m_tests_utils_test_seqlen_balancing --> m_verl_utils_seqlen_balancing
    m_tests_utils_test_server_profiler --> m_verl_utils_profiler_config
    m_tests_utils_test_server_profiler --> m_verl_workers_rollout_sglang_rollout_async_sglang_server
    m_tests_utils_test_server_profiler --> m_verl_workers_rollout_vllm_rollout_vllm_async_server
    m_tests_utils_test_shared_memory --> m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer
    m_tests_utils_test_special_linear_cross_entropy_tp --> m_verl_utils_distributed
    m_tests_utils_test_special_linear_cross_entropy_tp --> m_verl_utils_kernel_linear_cross_entropy
    m_tests_utils_test_special_linear_cross_entropy_tp --> m_verl_utils_torch_functional
    m_tests_utils_test_special_megatron_kl_loss_tp --> m_verl_trainer_config_config
    m_tests_utils_test_special_megatron_kl_loss_tp --> m_verl_trainer_distillation_fsdp_losses
    m_tests_utils_test_special_megatron_kl_loss_tp --> m_verl_trainer_distillation_megatron_losses
    m_tests_utils_test_special_megatron_kl_loss_tp --> m_verl_utils_distributed
    m_tests_utils_test_special_mstx_profile --> m_verl_utils_profiler_config
    m_tests_utils_test_special_mstx_profile --> m_verl_utils_profiler_mstx_profile
    m_tests_utils_test_special_mstx_profile --> m_verl_utils_profiler_profile
    m_tests_utils_test_temp_env_on_cpu --> m_verl_utils_py_functional
    m_tests_utils_test_timeout_decorator_cpu --> m_verl_utils_py_functional
    m_tests_utils_test_tokenizer_normalize_on_cpu --> m_verl_utils_tokenizer
    m_tests_utils_test_torch_functional --> m_verl_utils_device
    m_tests_utils_test_torch_functional --> m_verl_utils_distributed
    m_tests_utils_test_torch_functional --> m_verl_utils_torch_functional
    m_tests_utils_test_torch_profile --> m_verl_utils_profiler_config
    m_tests_utils_test_torch_profile --> m_verl_utils_profiler_torch_profile
    m_tests_workers_actor_test_special_dp_actor --> m_verl_trainer_config_config
    m_tests_workers_actor_test_special_dp_actor --> m_verl_utils_device
    m_tests_workers_actor_test_special_dp_actor --> m_verl_workers_actor_dp_actor
    m_tests_workers_config_test_actor_config_on_cpu --> m_verl_trainer_config_config
    m_tests_workers_config_test_actor_config_on_cpu --> m_verl_utils_config
    m_tests_workers_config_test_critic_config_on_cpu --> m_verl_trainer_config_config
    m_tests_workers_config_test_critic_config_on_cpu --> m_verl_utils_config
    m_tests_workers_config_test_engine_config_on_cpu --> m_verl_workers_config_engine
    m_tests_workers_config_test_model_config_on_cpu --> m_verl_workers_config_model
    m_tests_workers_config_test_optim_config_on_cpu --> m_verl_workers_config_optimizer
    m_tests_workers_critic_test_special_dp_critic --> m_verl_trainer_config_config
    m_tests_workers_critic_test_special_dp_critic --> m_verl_utils_distributed
    m_tests_workers_critic_test_special_dp_critic --> m_verl_workers_config_critic
    m_tests_workers_critic_test_special_dp_critic --> m_verl_workers_config_engine
    m_tests_workers_critic_test_special_dp_critic --> m_verl_workers_fsdp_workers
    m_tests_workers_reward_manager_test_registry_on_cpu --> m_verl_workers_reward_manager_registry
    m_tests_workers_rollout_perf_vllm_async_rollout --> m_my_recipes_rllm_utils
    m_tests_workers_rollout_perf_vllm_async_rollout --> m_my_recipes_syntool_dataset
    m_tests_workers_rollout_perf_vllm_async_rollout --> m_tests_experimental_agent_loop_agent_utils
    m_tests_workers_rollout_perf_vllm_async_rollout --> m_verl_protocol
    m_tests_workers_rollout_perf_vllm_async_rollout --> m_verl_utils_dataset_rl_dataset
    m_tests_workers_rollout_rollout_sglang_test_http_server_engine --> m_my_recipes_rllm_utils
    m_tests_workers_rollout_rollout_sglang_test_http_server_engine --> m_verl_workers_rollout_sglang_rollout_http_server_engine
    m_tests_workers_rollout_rollout_trtllm_test_adapter --> m_verl_workers_rollout_trtllm_rollout_trtllm_async_server
    m_tests_workers_rollout_rollout_trtllm_test_adapter --> m_verl_workers_rollout_trtllm_rollout_trtllm_rollout
    m_tests_workers_rollout_rollout_trtllm_test_async_server --> m_verl_models_mcore_util
    m_tests_workers_rollout_rollout_trtllm_test_async_server --> m_verl_workers_rollout_replica
    m_tests_workers_rollout_rollout_trtllm_test_async_server --> m_verl_workers_rollout_trtllm_rollout_trtllm_async_server
    m_tests_workers_rollout_rollout_trtllm_test_trtllm_rollout_utils --> m_verl_workers_rollout_trtllm_rollout_trtllm_async_server
    m_tests_workers_rollout_rollout_vllm_run_fsdp_vllm --> m_verl_utils_distributed
    m_tests_workers_rollout_rollout_vllm_run_fsdp_vllm --> m_verl_utils_fs
    m_tests_workers_rollout_rollout_vllm_run_fsdp_vllm --> m_verl_utils_torch_functional
    m_tests_workers_rollout_rollout_vllm_test_vllm_abort --> m_verl_utils_tokenizer
    m_tests_workers_rollout_rollout_vllm_test_vllm_abort --> m_verl_workers_rollout_replica
    m_tests_workers_rollout_rollout_vllm_test_vllm_omni_generate --> m_verl_utils_tokenizer
    m_tests_workers_rollout_rollout_vllm_test_vllm_omni_generate --> m_verl_workers_rollout_replica
    m_tests_workers_rollout_rollout_vllm_test_vllm_omni_generate --> m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server
    m_tests_workers_rollout_test_hf_rollout --> m_verl_utils_distributed
    m_tests_workers_rollout_test_hf_rollout --> m_verl_utils_fs
    m_tests_workers_rollout_test_hf_rollout --> m_verl_utils_model
    m_tests_workers_rollout_test_hf_rollout --> m_verl_workers_rollout_hf_rollout
    m_tests_workers_rollout_test_sglang_async_rollout_multimodal_delta --> m_verl_tools_schemas
    m_tests_workers_rollout_test_sglang_async_rollout_multimodal_delta --> m_verl_utils_dataset_vision_utils
    m_tests_workers_rollout_test_sglang_async_rollout_multimodal_delta --> m_verl_utils_tokenizer
    m_tests_workers_rollout_test_sglang_async_rollout_multimodal_delta --> m_verl_workers_rollout_schemas
    m_tests_workers_rollout_test_sglang_rollout_sharding_manager --> m_verl_workers_rollout_sglang_rollout_utils
    m_tests_workers_rollout_test_vllm_cli_args_on_cpu --> m_verl_workers_rollout_vllm_rollout_utils
    m_tests_workers_test_fsdp_attn_implementation --> m_verl_trainer_config_config
    m_tests_workers_test_fsdp_attn_implementation --> m_verl_workers_fsdp_workers
    m_tests_workers_test_fsdp_workers --> m_verl_workers_fsdp_workers
    m_verl___init__ --> m_verl_checkpoint_engine_base
    m_verl___init__ --> m_verl_protocol
    m_verl___init__ --> m_verl_utils_device
    m_verl___init__ --> m_verl_utils_import_utils
    m_verl___init__ --> m_verl_utils_logging_utils
    m_verl_checkpoint_engine___init__ --> m_verl_checkpoint_engine_base
    m_verl_checkpoint_engine___init__ --> m_verl_checkpoint_engine_hccl_checkpoint_engine
    m_verl_checkpoint_engine___init__ --> m_verl_checkpoint_engine_kimi_checkpoint_engine
    m_verl_checkpoint_engine___init__ --> m_verl_checkpoint_engine_mooncake_checkpoint_engine
    m_verl_checkpoint_engine___init__ --> m_verl_checkpoint_engine_nccl_checkpoint_engine
    m_verl_checkpoint_engine___init__ --> m_verl_checkpoint_engine_nixl_checkpoint_engine
    m_verl_checkpoint_engine_base --> m_verl_single_controller_base_decorator
    m_verl_checkpoint_engine_base --> m_verl_trainer_config_config
    m_verl_checkpoint_engine_base --> m_verl_utils_distributed
    m_verl_checkpoint_engine_base --> m_verl_utils_ray_utils
    m_verl_checkpoint_engine_base --> m_verl_workers_config_rollout
    m_verl_checkpoint_engine_hccl_checkpoint_engine --> m_my_recipes_rllm_utils
    m_verl_checkpoint_engine_hccl_checkpoint_engine --> m_verl_checkpoint_engine_base
    m_verl_checkpoint_engine_hccl_checkpoint_engine --> m_verl_utils_device
    m_verl_checkpoint_engine_hccl_checkpoint_engine --> m_verl_utils_distributed
    m_verl_checkpoint_engine_hccl_checkpoint_engine --> m_verl_utils_net_utils
    m_verl_checkpoint_engine_kimi_checkpoint_engine --> m_verl_checkpoint_engine_base
    m_verl_checkpoint_engine_kimi_checkpoint_engine --> m_verl_utils_device
    m_verl_checkpoint_engine_kimi_checkpoint_engine --> m_verl_utils_distributed
    m_verl_checkpoint_engine_kimi_checkpoint_engine --> m_verl_utils_net_utils
    m_verl_checkpoint_engine_mooncake_checkpoint_engine --> m_my_recipes_rllm_utils
    m_verl_checkpoint_engine_mooncake_checkpoint_engine --> m_verl_checkpoint_engine_base
    m_verl_checkpoint_engine_mooncake_checkpoint_engine --> m_verl_utils_device
    m_verl_checkpoint_engine_mooncake_checkpoint_engine --> m_verl_utils_net_utils
    m_verl_checkpoint_engine_mooncake_checkpoint_engine --> m_verl_workers_config_engine
    m_verl_checkpoint_engine_nccl_checkpoint_engine --> m_verl_checkpoint_engine_base
    m_verl_checkpoint_engine_nccl_checkpoint_engine --> m_verl_utils_net_utils
    m_verl_checkpoint_engine_nixl_checkpoint_engine --> m_verl_checkpoint_engine_base
    m_verl_checkpoint_engine_nixl_checkpoint_engine --> m_verl_utils_net_utils
    m_verl_experimental_agent_loop___init__ --> m_my_recipes_syntool_agent_loop
    m_verl_experimental_agent_loop___init__ --> m_verl_experimental_agent_loop_single_turn_agent_loop
    m_verl_experimental_agent_loop___init__ --> m_verl_experimental_agent_loop_tool_agent_loop
    m_verl_experimental_agent_loop_agent_loop --> m_verl_experimental_agent_loop_prometheus_utils
    m_verl_experimental_agent_loop_agent_loop --> m_verl_experimental_agent_loop_utils
    m_verl_experimental_agent_loop_agent_loop --> m_verl_experimental_teacher_loop_teacher_manager
    m_verl_experimental_agent_loop_agent_loop --> m_verl_protocol
    m_verl_experimental_agent_loop_agent_loop --> m_verl_single_controller_ray_base
    m_verl_experimental_agent_loop_agent_loop --> m_verl_trainer_config_config
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_chat_template
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_config
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_dataset_rl_dataset
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_model
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_ray_utils
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_rollout_trace
    m_verl_experimental_agent_loop_agent_loop --> m_verl_utils_tokenizer
    m_verl_experimental_agent_loop_agent_loop --> m_verl_workers_config_distillation
    m_verl_experimental_agent_loop_agent_loop --> m_verl_workers_rollout_replica
    m_verl_experimental_agent_loop_prometheus_utils --> m_verl_workers_config_rollout
    m_verl_experimental_agent_loop_single_turn_agent_loop --> m_verl_experimental_agent_loop_agent_loop
    m_verl_experimental_agent_loop_single_turn_agent_loop --> m_verl_workers_rollout_replica
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_experimental_agent_loop_agent_loop
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_experimental_agent_loop_tool_parser
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_experimental_agent_loop_utils
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_interactions_base
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_interactions_utils_interaction_registry
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_tools_schemas
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_tools_utils_tool_registry
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_utils_rollout_trace
    m_verl_experimental_agent_loop_tool_agent_loop --> m_verl_workers_rollout_replica
    m_verl_experimental_agent_loop_tool_parser --> m_verl_tools_schemas
    m_verl_experimental_agent_loop_tool_parser --> m_verl_utils_ray_utils
    m_verl_experimental_agent_loop_tool_parser --> m_verl_utils_rollout_trace
    m_verl_experimental_dynamic_dataset_dynamicgen_dataset --> m_my_recipes_syntool_dataset
    m_verl_experimental_dynamic_dataset_dynamicgen_dataset --> m_verl_utils_import_utils
    m_verl_experimental_fully_async_policy_agent_loop___init__ --> m_my_recipes_syntool_agent_loop
    m_verl_experimental_fully_async_policy_agent_loop_agent_loop --> m_verl_experimental_agent_loop_agent_loop
    m_verl_experimental_fully_async_policy_agent_loop_agent_loop --> m_verl_protocol
    m_verl_experimental_fully_async_policy_agent_loop_agent_loop --> m_verl_utils_ray_utils
    m_verl_experimental_fully_async_policy_agent_loop_agent_loop --> m_verl_utils_rollout_trace
    m_verl_experimental_fully_async_policy_detach_utils --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_fully_async_policy_fully_async_main --> m_my_recipes_rllm_utils
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_experimental_fully_async_policy_fully_async_rollouter
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_experimental_fully_async_policy_fully_async_trainer
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_experimental_fully_async_policy_message_queue
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_experimental_separation_utils
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_trainer_main_ppo
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_trainer_ppo_utils
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_utils_device
    m_verl_experimental_fully_async_policy_fully_async_main --> m_verl_utils_fs
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_my_recipes_syntool_agent_loop
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_experimental_fully_async_policy_detach_utils
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_experimental_fully_async_policy_message_queue
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_experimental_separation_ray_trainer
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_trainer_main_ppo
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_trainer_ppo_utils
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_utils_dataset_rl_dataset
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_utils_fs
    m_verl_experimental_fully_async_policy_fully_async_rollouter --> m_verl_utils_tracking
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_my_recipes_syntool_agent_loop
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_experimental_fully_async_policy_detach_utils
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_experimental_fully_async_policy_message_queue
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_experimental_separation_ray_trainer
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_trainer_main_ppo
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_trainer_ppo_utils
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_utils_config
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_utils_dataset_rl_dataset
    m_verl_experimental_fully_async_policy_fully_async_trainer --> m_verl_utils_tracking
    m_verl_experimental_one_step_off_policy_main_ppo --> m_my_recipes_rllm_utils
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_experimental_one_step_off_policy_ray_trainer
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_experimental_separation_utils
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_trainer_main_ppo
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_trainer_ppo_utils
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_utils_config
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_utils_dataset_rl_dataset
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_utils_device
    m_verl_experimental_one_step_off_policy_main_ppo --> m_verl_utils_fs
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_my_recipes_syntool_agent_loop
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_verl_experimental_separation_ray_trainer
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_verl_trainer_ppo_reward
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_verl_trainer_ppo_utils
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_verl_utils_rollout_skip
    m_verl_experimental_one_step_off_policy_ray_trainer --> m_verl_utils_tracking
    m_verl_experimental_reward_loop___init__ --> m_verl_experimental_reward_loop_reward_loop
    m_verl_experimental_reward_loop___init__ --> m_verl_experimental_reward_loop_reward_model
    m_verl_experimental_reward_loop_reward_loop --> m_my_recipes_rllm_utils
    m_verl_experimental_reward_loop_reward_loop --> m_verl_experimental_reward_loop_reward_model
    m_verl_experimental_reward_loop_reward_loop --> m_verl_protocol
    m_verl_experimental_reward_loop_reward_loop --> m_verl_single_controller_ray_base
    m_verl_experimental_reward_loop_reward_loop --> m_verl_trainer_ppo_reward
    m_verl_experimental_reward_loop_reward_loop --> m_verl_utils_experimental_reward_utils
    m_verl_experimental_reward_loop_reward_loop --> m_verl_utils_fs
    m_verl_experimental_reward_loop_reward_loop --> m_verl_utils_ray_utils
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_dapo
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_gdpo
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_limited
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_naive
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_registry
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_remote
    m_verl_experimental_reward_loop_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_visual
    m_verl_experimental_reward_loop_reward_manager_base --> m_verl_utils_ray_utils
    m_verl_experimental_reward_loop_reward_manager_dapo --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_manager_gdpo --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_manager_limited --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_manager_limited --> m_verl_utils_ray_utils
    m_verl_experimental_reward_loop_reward_manager_naive --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_manager_registry --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_manager_remote --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_manager_visual --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_experimental_reward_loop_reward_model --> m_verl_experimental_reward_loop_router_naive_router
    m_verl_experimental_reward_loop_reward_model --> m_verl_single_controller_ray_base
    m_verl_experimental_reward_loop_reward_model --> m_verl_trainer_config_config
    m_verl_experimental_reward_loop_reward_model --> m_verl_workers_rollout_replica
    m_verl_experimental_reward_loop_router_inner_sglang_router --> m_verl_utils_net_utils
    m_verl_experimental_reward_loop_router_naive_router --> m_verl_utils_net_utils
    m_verl_experimental_separation_engine_workers --> m_verl_single_controller_base_decorator
    m_verl_experimental_separation_engine_workers --> m_verl_utils_device
    m_verl_experimental_separation_engine_workers --> m_verl_utils_fsdp_utils
    m_verl_experimental_separation_engine_workers --> m_verl_utils_megatron_utils
    m_verl_experimental_separation_engine_workers --> m_verl_workers_engine_workers
    m_verl_experimental_separation_ray_trainer --> m_verl_experimental_dataset_sampler
    m_verl_experimental_separation_ray_trainer --> m_verl_experimental_reward_loop_reward_loop
    m_verl_experimental_separation_ray_trainer --> m_verl_single_controller_ray_base
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_config_config
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_ppo_core_algos
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_ppo_metric_utils
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_ppo_reward
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_ppo_rollout_corr_helper
    m_verl_experimental_separation_ray_trainer --> m_verl_trainer_ppo_utils
    m_verl_experimental_separation_ray_trainer --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_experimental_separation_ray_trainer --> m_verl_utils_config
    m_verl_experimental_separation_ray_trainer --> m_verl_utils_debug_metrics
    m_verl_experimental_separation_ray_trainer --> m_verl_utils_rollout_skip
    m_verl_experimental_separation_ray_trainer --> m_verl_utils_tracking
    m_verl_experimental_separation_ray_trainer --> m_verl_workers_engine_workers
    m_verl_experimental_separation_ray_trainer --> m_verl_workers_utils_losses
    m_verl_experimental_separation_utils --> m_verl_experimental_separation_engine_workers
    m_verl_experimental_separation_utils --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_separation_utils --> m_verl_trainer_ppo_utils
    m_verl_experimental_separation_utils --> m_verl_workers_engine_workers
    m_verl_experimental_teacher_loop___init__ --> m_verl_experimental_teacher_loop_teacher_model
    m_verl_experimental_teacher_loop_teacher_manager --> m_my_recipes_syntool_agent_loop
    m_verl_experimental_teacher_loop_teacher_manager --> m_verl_protocol
    m_verl_experimental_teacher_loop_teacher_manager --> m_verl_trainer_config_config
    m_verl_experimental_teacher_loop_teacher_manager --> m_verl_utils_config
    m_verl_experimental_teacher_loop_teacher_manager --> m_verl_utils_tokenizer
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_experimental_agent_loop_agent_loop
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_experimental_reward_loop_router_naive_router
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_experimental_teacher_loop_teacher_manager
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_single_controller_ray_base
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_trainer_config_config
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_utils_config
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_utils_ray_utils
    m_verl_experimental_teacher_loop_teacher_model --> m_verl_workers_rollout_replica
    m_verl_experimental_vla_dp_rob --> m_verl_checkpoint_engine_base
    m_verl_experimental_vla_dp_rob --> m_verl_protocol
    m_verl_experimental_vla_dp_rob --> m_verl_utils_device
    m_verl_experimental_vla_dp_rob --> m_verl_utils_py_functional
    m_verl_experimental_vla_dp_rob --> m_verl_utils_seqlen_balancing
    m_verl_experimental_vla_dp_rob --> m_verl_utils_torch_functional
    m_verl_experimental_vla_dp_rob --> m_verl_workers_config_actor
    m_verl_experimental_vla_envs_action_utils --> m_verl_experimental_vla_envs_libero_env_utils
    m_verl_experimental_vla_envs_isaac_env___init__ --> m_verl_experimental_vla_envs_isaac_env_isaac_env
    m_verl_experimental_vla_envs_isaac_env_isaac_env --> m_my_recipes_rllm_utils
    m_verl_experimental_vla_envs_isaac_env_isaac_env --> m_verl_experimental_vla_envs_action_utils
    m_verl_experimental_vla_envs_libero_env_libero_env --> m_verl_experimental_vla_envs_action_utils
    m_verl_experimental_vla_envs_libero_env_libero_env --> m_verl_experimental_vla_envs_libero_env_utils
    m_verl_experimental_vla_envs_libero_env_libero_env --> m_verl_experimental_vla_envs_libero_env_venv
    m_verl_experimental_vla_fsdp_workers --> m_verl_experimental_vla_dp_rob
    m_verl_experimental_vla_fsdp_workers --> m_verl_experimental_vla_naive_rollout_rob
    m_verl_experimental_vla_fsdp_workers --> m_verl_experimental_vla_sac_naive_rollout_pi05
    m_verl_experimental_vla_fsdp_workers --> m_verl_experimental_vla_sac_sac_actor
    m_verl_experimental_vla_fsdp_workers --> m_verl_single_controller_base_decorator
    m_verl_experimental_vla_fsdp_workers --> m_verl_trainer_config_config
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_checkpoint_fsdp_checkpoint_manager
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_config
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_device
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_distributed
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_flops_counter
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_fsdp_utils
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_import_utils
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_memory_utils
    m_verl_experimental_vla_fsdp_workers --> m_verl_utils_profiler_performance
    m_verl_experimental_vla_fsdp_workers --> m_verl_workers_fsdp_workers
    m_verl_experimental_vla_main_ppo --> m_my_recipes_rllm_utils
    m_verl_experimental_vla_main_ppo --> m_verl_experimental_vla_fsdp_workers
    m_verl_experimental_vla_main_ppo --> m_verl_experimental_vla_rob_ray_trainer
    m_verl_experimental_vla_main_ppo --> m_verl_experimental_vla_workers_env_env_worker
    m_verl_experimental_vla_main_ppo --> m_verl_trainer_constants_ppo
    m_verl_experimental_vla_main_ppo --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_vla_main_ppo --> m_verl_trainer_ppo_utils
    m_verl_experimental_vla_main_ppo --> m_verl_utils_device
    m_verl_experimental_vla_main_ppo --> m_verl_utils_fs
    m_verl_experimental_vla_main_ppo --> m_verl_utils_import_utils
    m_verl_experimental_vla_main_sac --> m_my_recipes_rllm_utils
    m_verl_experimental_vla_main_sac --> m_verl_experimental_vla_fsdp_workers
    m_verl_experimental_vla_main_sac --> m_verl_experimental_vla_sac_sac_ray_trainer
    m_verl_experimental_vla_main_sac --> m_verl_experimental_vla_workers_env_env_worker
    m_verl_experimental_vla_main_sac --> m_verl_trainer_constants_ppo
    m_verl_experimental_vla_main_sac --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_vla_main_sac --> m_verl_trainer_ppo_utils
    m_verl_experimental_vla_main_sac --> m_verl_utils_fs
    m_verl_experimental_vla_models___init__ --> m_verl_experimental_vla_models_register_vla_models
    m_verl_experimental_vla_models_openvla_oft_modeling_prismatic --> m_verl_experimental_vla_models_openvla_oft_configuration_prismatic
    m_verl_experimental_vla_models_openvla_oft_modeling_prismatic --> m_verl_experimental_vla_models_openvla_oft_constants
    m_verl_experimental_vla_models_openvla_oft_modeling_prismatic --> m_verl_experimental_vla_models_openvla_oft_train_utils
    m_verl_experimental_vla_models_openvla_oft_processing_prismatic --> m_my_recipes_rllm_utils
    m_verl_experimental_vla_models_openvla_oft_train_utils --> m_verl_experimental_vla_models_openvla_oft_constants
    m_verl_experimental_vla_models_pi0_torch___init__ --> m_verl_experimental_vla_models_pi0_torch_configuration_pi0_torch
    m_verl_experimental_vla_models_pi0_torch___init__ --> m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch
    m_verl_experimental_vla_models_pi0_torch_model_modeling_pi0 --> m_verl_experimental_vla_models_pi0_torch_model_paligemma_with_expert
    m_verl_experimental_vla_models_pi0_torch_model_paligemma_with_expert --> m_my_recipes_rllm_utils
    m_verl_experimental_vla_models_pi0_torch_model_paligemma_with_expert --> m_verl_utils_device
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_checkpoint_engine_base
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_experimental_vla_models_modules_mlp
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_experimental_vla_models_pi0_torch_configuration_pi0_torch
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_experimental_vla_models_pi0_torch_model_modeling_pi0
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_experimental_vla_models_pi0_torch_pi0_utils
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_experimental_vla_models_pi0_torch_policy_libero_policy
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_protocol
    m_verl_experimental_vla_models_pi0_torch_modeling_pi0_torch --> m_verl_utils_device
    m_verl_experimental_vla_models_pi0_torch_policy_libero_policy --> m_verl_checkpoint_engine_base
    m_verl_experimental_vla_models_pi0_torch_policy_libero_policy --> m_verl_protocol
    m_verl_experimental_vla_models_register_vla_models --> m_verl_experimental_vla_models_openvla_oft_configuration_prismatic
    m_verl_experimental_vla_models_register_vla_models --> m_verl_experimental_vla_models_openvla_oft_modeling_prismatic
    m_verl_experimental_vla_models_register_vla_models --> m_verl_experimental_vla_models_openvla_oft_processing_prismatic
    m_verl_experimental_vla_models_register_vla_models --> m_verl_utils_transformers_compat
    m_verl_experimental_vla_naive_rollout_rob --> m_verl_experimental_vla_envs_action_utils
    m_verl_experimental_vla_naive_rollout_rob --> m_verl_experimental_vla_models_openvla_oft_modeling_prismatic
    m_verl_experimental_vla_naive_rollout_rob --> m_verl_experimental_vla_models_openvla_oft_processing_prismatic
    m_verl_experimental_vla_naive_rollout_rob --> m_verl_utils_device
    m_verl_experimental_vla_naive_rollout_rob --> m_verl_workers_rollout_base
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_experimental_dataset_sampler
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_experimental_vla_env_loop
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_protocol
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_single_controller_ray_base
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_trainer_ppo_core_algos
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_trainer_ppo_metric_utils
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_trainer_ppo_reward
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_trainer_ppo_utils
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_utils_debug_metrics
    m_verl_experimental_vla_rob_ray_trainer --> m_verl_utils_tracking
    m_verl_experimental_vla_sac_naive_rollout_pi05 --> m_verl_experimental_vla_naive_rollout_rob
    m_verl_experimental_vla_sac_naive_rollout_pi05 --> m_verl_utils_device
    m_verl_experimental_vla_sac_sac_actor --> m_verl_checkpoint_engine_base
    m_verl_experimental_vla_sac_sac_actor --> m_verl_experimental_vla_sac_replay_pool
    m_verl_experimental_vla_sac_sac_actor --> m_verl_protocol
    m_verl_experimental_vla_sac_sac_actor --> m_verl_utils_device
    m_verl_experimental_vla_sac_sac_ray_trainer --> m_verl_experimental_vla_env_loop
    m_verl_experimental_vla_sac_sac_ray_trainer --> m_verl_single_controller_ray_base
    m_verl_experimental_vla_sac_sac_ray_trainer --> m_verl_trainer_ppo_ray_trainer
    m_verl_experimental_vla_sac_sac_ray_trainer --> m_verl_trainer_ppo_utils
    m_verl_experimental_vla_sac_sac_ray_trainer --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_experimental_vla_sac_sac_ray_trainer --> m_verl_utils_tracking
    m_verl_experimental_vla_workers_env_env_loop_wg_test --> m_verl_experimental_vla_naive_rollout_rob
    m_verl_experimental_vla_workers_env_env_loop_wg_test --> m_verl_experimental_vla_workers_env_env_worker
    m_verl_experimental_vla_workers_env_env_loop_wg_test --> m_verl_single_controller_ray_base
    m_verl_experimental_vla_workers_env_env_manager --> m_verl_utils_device
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_checkpoint_engine_base
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_experimental_vla_envs_isaac_env_isaac_env
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_experimental_vla_envs_libero_env_libero_env
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_experimental_vla_workers_env_env_manager
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_single_controller_base_decorator
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_utils_config
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_utils_device
    m_verl_experimental_vla_workers_env_env_worker --> m_verl_utils_distributed
    m_verl_interactions_gsm8k_interaction --> m_verl_checkpoint_engine_base
    m_verl_interactions_gym_interaction --> m_verl_checkpoint_engine_base
    m_verl_interactions_gym_interaction --> m_verl_interactions_gym_env
    m_verl_interactions_utils_interaction_registry --> m_verl_models_mcore_util
    m_verl_interactions_weather_interaction --> m_verl_checkpoint_engine_base
    m_verl_model_merger___main__ --> m_verl_model_merger_base_model_merger
    m_verl_model_merger___main__ --> m_verl_model_merger_fsdp_model_merger
    m_verl_model_merger___main__ --> m_verl_model_merger_megatron_model_merger
    m_verl_model_merger_base_model_merger --> m_my_recipes_rllm_utils
    m_verl_model_merger_base_model_merger --> m_verl_utils_transformers_compat
    m_verl_model_merger_fsdp_model_merger --> m_verl_model_merger_base_model_merger
    m_verl_model_merger_megatron_model_merger --> m_verl_model_merger_base_model_merger
    m_verl_model_merger_megatron_model_merger --> m_verl_utils_device
    m_verl_model_merger_megatron_model_merger --> m_verl_utils_distributed
    m_verl_model_merger_megatron_model_merger --> m_verl_utils_megatron_dist_checkpointing
    m_verl_model_merger_megatron_model_merger --> m_verl_utils_megatron_utils
    m_verl_model_merger_megatron_model_merger --> m_verl_utils_qat_core
    m_verl_model_merger_megatron_model_merger --> m_verl_utils_tokenizer
    m_verl_models_mcore___init__ --> m_verl_experimental_reward_loop_reward_manager_registry
    m_verl_models_mcore___init__ --> m_verl_models_mcore_patch
    m_verl_models_mcore_bridge --> m_verl_utils_qat_core
    m_verl_models_mcore_config_converter --> m_verl_models_mcore_patch
    m_verl_models_mcore_config_converter --> m_verl_trainer_config_config
    m_verl_models_mcore_config_converter --> m_verl_utils_qat_core
    m_verl_models_mcore_loader --> m_verl_models_mcore_saver
    m_verl_models_mcore_loader --> m_verl_utils_device
    m_verl_models_mcore_loader --> m_verl_utils_distributed
    m_verl_models_mcore_loader --> m_verl_utils_megatron_utils
    m_verl_models_mcore_loader --> m_verl_utils_qat_core
    m_verl_models_mcore_mbridge --> m_verl_models_mcore_patch
    m_verl_models_mcore_model_forward --> m_verl_models_mcore_util
    m_verl_models_mcore_model_forward --> m_verl_trainer_config_config
    m_verl_models_mcore_model_forward --> m_verl_utils_megatron_utils
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_my_recipes_rllm_utils
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_verl_models_mcore_util
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_verl_utils_kernel_linear_cross_entropy
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_verl_utils_megatron_tensor_parallel
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_verl_utils_megatron_utils
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_verl_utils_model
    m_verl_models_mcore_model_forward_1f1b_overlap --> m_verl_utils_qat_core
    m_verl_models_mcore_model_forward_fused --> m_my_recipes_rllm_utils
    m_verl_models_mcore_model_forward_fused --> m_verl_models_mcore_util
    m_verl_models_mcore_model_forward_fused --> m_verl_utils_kernel_linear_cross_entropy
    m_verl_models_mcore_model_forward_fused --> m_verl_utils_megatron_utils
    m_verl_models_mcore_model_forward_fused --> m_verl_utils_model
    m_verl_models_mcore_model_forward_fused --> m_verl_utils_qat_core
    m_verl_models_mcore_model_initializer --> m_verl_models_mcore_bridge
    m_verl_models_mcore_model_initializer --> m_verl_models_mcore_config_converter
    m_verl_models_mcore_mtp_patch --> m_my_recipes_rllm_utils
    m_verl_models_mcore_mtp_patch --> m_verl_utils_megatron_utils
    m_verl_models_mcore_mtp_patch --> m_verl_utils_qat_core
    m_verl_models_mcore_patch --> m_my_recipes_rllm_utils
    m_verl_models_mcore_patch --> m_verl_utils_qat_core
    m_verl_models_mcore_qwen2_5_vl___init__ --> m_verl_models_mcore_qwen2_5_vl_model
    m_verl_models_mcore_qwen2_5_vl___init__ --> m_verl_models_mcore_qwen2_5_vl_vision_config
    m_verl_models_mcore_qwen2_5_vl_attention --> m_verl_models_mcore_qwen2_5_vl_rope_utils
    m_verl_models_mcore_qwen2_5_vl_model --> m_verl_models_mcore_qwen2_5_vl_attention
    m_verl_models_mcore_qwen2_5_vl_model --> m_verl_models_mcore_qwen2_5_vl_rope_utils
    m_verl_models_mcore_qwen2_5_vl_model --> m_verl_models_mcore_qwen2_5_vl_vision_model
    m_verl_models_mcore_qwen2_5_vl_model --> m_verl_models_mcore_util
    m_verl_models_mcore_qwen2_5_vl_model --> m_verl_utils_qat_core
    m_verl_models_mcore_qwen2_5_vl_vision_config --> m_verl_utils_qat_core
    m_verl_models_mcore_qwen2_5_vl_vision_model --> m_verl_models_mcore_qwen2_5_vl_vision_transformer_block
    m_verl_models_mcore_qwen2_5_vl_vision_model --> m_verl_utils_qat_core
    m_verl_models_mcore_registry --> m_verl_models_mcore_config_converter
    m_verl_models_mcore_registry --> m_verl_models_mcore_model_forward
    m_verl_models_mcore_registry --> m_verl_models_mcore_model_forward_fused
    m_verl_models_mcore_registry --> m_verl_models_mcore_model_initializer
    m_verl_models_mcore_registry --> m_verl_models_mcore_weight_converter
    m_verl_models_mcore_saver --> m_verl_utils_device
    m_verl_models_mcore_saver --> m_verl_utils_distributed
    m_verl_models_mcore_saver --> m_verl_utils_megatron_utils
    m_verl_models_mcore_saver --> m_verl_utils_qat_core
    m_verl_models_mcore_util --> m_verl_utils_device
    m_verl_models_mcore_util --> m_verl_utils_model
    m_verl_models_mcore_util --> m_verl_utils_qat_core
    m_verl_models_transformers___init__ --> m_verl_models_transformers_monkey_patch
    m_verl_models_transformers___init__ --> m_verl_models_transformers_tiled_mlp
    m_verl_models_transformers_apertus --> m_my_recipes_rllm_utils
    m_verl_models_transformers_apertus --> m_verl_utils_ulysses
    m_verl_models_transformers_dense_common --> m_verl_utils_experimental_torch_functional
    m_verl_models_transformers_dense_common --> m_verl_utils_kernel_linear_cross_entropy
    m_verl_models_transformers_glm4v --> m_my_recipes_rllm_utils
    m_verl_models_transformers_glm4v --> m_verl_utils_device
    m_verl_models_transformers_glm4v --> m_verl_utils_distributed
    m_verl_models_transformers_glm4v --> m_verl_utils_experimental_torch_functional
    m_verl_models_transformers_glm4v --> m_verl_utils_kernel_linear_cross_entropy
    m_verl_models_transformers_glm4v --> m_verl_utils_ulysses
    m_verl_models_transformers_kimi_vl --> m_verl_models_transformers_monkey_patch
    m_verl_models_transformers_kimi_vl --> m_verl_utils_transformers_compat
    m_verl_models_transformers_kimi_vl --> m_verl_utils_ulysses
    m_verl_models_transformers_llama --> m_my_recipes_rllm_utils
    m_verl_models_transformers_llama --> m_verl_utils_transformers_compat
    m_verl_models_transformers_llama --> m_verl_utils_ulysses
    m_verl_models_transformers_monkey_patch --> m_verl_models_transformers_dense_common
    m_verl_models_transformers_monkey_patch --> m_verl_models_transformers_glm4v
    m_verl_models_transformers_monkey_patch --> m_verl_models_transformers_kimi_vl
    m_verl_models_transformers_monkey_patch --> m_verl_models_transformers_qwen2_vl
    m_verl_models_transformers_monkey_patch --> m_verl_models_transformers_qwen3_vl
    m_verl_models_transformers_monkey_patch --> m_verl_models_transformers_tiled_mlp
    m_verl_models_transformers_monkey_patch --> m_verl_utils_import_utils
    m_verl_models_transformers_monkey_patch --> m_verl_utils_transformers_compat
    m_verl_models_transformers_monkey_patch --> m_verl_utils_ulysses
    m_verl_models_transformers_npu_patch --> m_my_recipes_rllm_utils
    m_verl_models_transformers_npu_patch --> m_verl_models_transformers_qwen2
    m_verl_models_transformers_npu_patch --> m_verl_models_transformers_qwen3_vl
    m_verl_models_transformers_qwen2 --> m_my_recipes_rllm_utils
    m_verl_models_transformers_qwen2 --> m_verl_utils_transformers_compat
    m_verl_models_transformers_qwen2 --> m_verl_utils_ulysses
    m_verl_models_transformers_qwen2_vl --> m_my_recipes_rllm_utils
    m_verl_models_transformers_qwen2_vl --> m_verl_utils_device
    m_verl_models_transformers_qwen2_vl --> m_verl_utils_distributed
    m_verl_models_transformers_qwen2_vl --> m_verl_utils_experimental_torch_functional
    m_verl_models_transformers_qwen2_vl --> m_verl_utils_kernel_linear_cross_entropy
    m_verl_models_transformers_qwen2_vl --> m_verl_utils_transformers_compat
    m_verl_models_transformers_qwen2_vl --> m_verl_utils_ulysses
    m_verl_models_transformers_qwen3_vl --> m_verl_utils_experimental_torch_functional
    m_verl_models_transformers_qwen3_vl --> m_verl_utils_kernel_linear_cross_entropy
    m_verl_models_weight_loader_registry --> m_verl_models_mcore_loader
    m_verl_models_weight_loader_registry --> m_verl_models_mcore_saver
    m_verl_protocol --> m_my_recipes_rllm_utils
    m_verl_protocol --> m_verl_utils_device
    m_verl_protocol --> m_verl_utils_distributed
    m_verl_protocol --> m_verl_utils_py_functional
    m_verl_protocol --> m_verl_utils_tensordict_utils
    m_verl_protocol --> m_verl_utils_torch_functional
    m_verl_single_controller___init__ --> m_verl_checkpoint_engine_base
    m_verl_single_controller_base___init__ --> m_verl_single_controller_base_worker
    m_verl_single_controller_base___init__ --> m_verl_single_controller_base_worker_group
    m_verl_single_controller_base_decorator --> m_verl_protocol
    m_verl_single_controller_base_decorator --> m_verl_single_controller_base_worker_group
    m_verl_single_controller_base_decorator --> m_verl_utils_py_functional
    m_verl_single_controller_base_decorator --> m_verl_utils_ray_utils
    m_verl_single_controller_base_worker --> m_verl_single_controller_base_decorator
    m_verl_single_controller_base_worker --> m_verl_utils_device
    m_verl_single_controller_base_worker --> m_verl_utils_ray_utils
    m_verl_single_controller_base_worker_group --> m_verl_single_controller_base_decorator
    m_verl_single_controller_ray___init__ --> m_verl_checkpoint_engine_base
    m_verl_single_controller_ray_base --> m_verl_checkpoint_engine_base
    m_verl_single_controller_ray_base --> m_verl_protocol
    m_verl_single_controller_ray_base --> m_verl_single_controller_base_decorator
    m_verl_single_controller_ray_base --> m_verl_utils_device
    m_verl_single_controller_ray_base --> m_verl_utils_py_functional
    m_verl_third_party_torch_distributed__state_dict_utils --> m_verl_utils_distributed
    m_verl_third_party_torch_distributed_checkpoint_state_dict --> m_verl_third_party_torch_distributed__state_dict_utils
    m_verl_third_party_torch_distributed_checkpoint_state_dict --> m_verl_utils_distributed
    m_verl_third_party_vllm___init__ --> m_verl_utils_device
    m_verl_third_party_vllm___init__ --> m_verl_utils_distributed
    m_verl_third_party_vllm___init__ --> m_verl_utils_import_utils
    m_verl_tools_base_tool --> m_verl_tools_schemas
    m_verl_tools_base_tool --> m_verl_utils_rollout_trace
    m_verl_tools_geo3k_tool --> m_verl_tools_base_tool
    m_verl_tools_geo3k_tool --> m_verl_tools_schemas
    m_verl_tools_geo3k_tool --> m_verl_utils_rollout_trace
    m_verl_tools_gsm8k_tool --> m_verl_tools_base_tool
    m_verl_tools_gsm8k_tool --> m_verl_tools_schemas
    m_verl_tools_gsm8k_tool --> m_verl_utils_rollout_trace
    m_verl_tools_image_zoom_in_tool --> m_verl_tools_base_tool
    m_verl_tools_image_zoom_in_tool --> m_verl_tools_schemas
    m_verl_tools_image_zoom_in_tool --> m_verl_workers_config_actor
    m_verl_tools_mcp_base_tool --> m_verl_tools_base_tool
    m_verl_tools_mcp_base_tool --> m_verl_tools_schemas
    m_verl_tools_mcp_base_tool --> m_verl_tools_utils_mcp_clients_McpClientManager
    m_verl_tools_mcp_base_tool --> m_verl_utils_rollout_trace
    m_verl_tools_mcp_search_tool --> m_verl_tools_mcp_base_tool
    m_verl_tools_mcp_search_tool --> m_verl_tools_schemas
    m_verl_tools_sandbox_fusion_tools --> m_verl_tools_base_tool
    m_verl_tools_sandbox_fusion_tools --> m_verl_tools_schemas
    m_verl_tools_sandbox_fusion_tools --> m_verl_utils_reward_score_sandbox_fusion_utils
    m_verl_tools_sandbox_fusion_tools --> m_verl_utils_rollout_trace
    m_verl_tools_search_tool --> m_verl_tools_base_tool
    m_verl_tools_search_tool --> m_verl_tools_schemas
    m_verl_tools_search_tool --> m_verl_tools_utils_search_r1_like_utils
    m_verl_tools_search_tool --> m_verl_utils_rollout_trace
    m_verl_tools_search_tool --> m_verl_workers_config_actor
    m_verl_tools_utils_mcp_clients_McpClientManager --> m_verl_tools_utils_mcp_clients_utils
    m_verl_tools_utils_tool_registry --> m_verl_tools_schemas
    m_verl_tools_utils_tool_registry --> m_verl_tools_utils_mcp_clients_McpClientManager
    m_verl_trainer_config___init__ --> m_verl_trainer_config_algorithm
    m_verl_trainer_config___init__ --> m_verl_trainer_config_config
    m_verl_trainer_config_algorithm --> m_verl_base_config
    m_verl_trainer_config_config --> m_verl_base_config
    m_verl_trainer_constants_ppo --> m_verl_experimental_vla_models_openvla_oft_constants
    m_verl_trainer_distillation___init__ --> m_verl_trainer_distillation_losses
    m_verl_trainer_distillation_fsdp_losses --> m_verl_trainer_config_config
    m_verl_trainer_distillation_fsdp_losses --> m_verl_utils_ulysses
    m_verl_trainer_distillation_losses --> m_verl_base_config
    m_verl_trainer_distillation_losses --> m_verl_trainer_config_config
    m_verl_trainer_distillation_losses --> m_verl_trainer_distillation_fsdp_losses
    m_verl_trainer_distillation_losses --> m_verl_trainer_distillation_megatron_losses
    m_verl_trainer_distillation_losses --> m_verl_trainer_ppo_core_algos
    m_verl_trainer_distillation_losses --> m_verl_workers_utils_losses
    m_verl_trainer_distillation_losses --> m_verl_workers_utils_padding
    m_verl_trainer_distillation_megatron_losses --> m_my_recipes_rllm_utils
    m_verl_trainer_distillation_megatron_losses --> m_verl_models_mcore_util
    m_verl_trainer_distillation_megatron_losses --> m_verl_trainer_config_config
    m_verl_trainer_main_eval --> m_verl_trainer_ppo_reward
    m_verl_trainer_main_eval --> m_verl_utils_fs
    m_verl_trainer_main_generation_server --> m_verl_utils_hdfs_io
    m_verl_trainer_main_generation_server --> m_verl_workers_rollout_replica
    m_verl_trainer_main_ppo --> m_my_recipes_rllm_utils
    m_verl_trainer_main_ppo --> m_verl_experimental_dataset_sampler
    m_verl_trainer_main_ppo --> m_verl_experimental_reward_loop_reward_loop
    m_verl_trainer_main_ppo --> m_verl_trainer_constants_ppo
    m_verl_trainer_main_ppo --> m_verl_trainer_ppo_ray_trainer
    m_verl_trainer_main_ppo --> m_verl_trainer_ppo_utils
    m_verl_trainer_main_ppo --> m_verl_utils_config
    m_verl_trainer_main_ppo --> m_verl_utils_dataset_rl_dataset
    m_verl_trainer_main_ppo --> m_verl_utils_device
    m_verl_trainer_main_ppo --> m_verl_utils_fs
    m_verl_trainer_main_ppo --> m_verl_utils_import_utils
    m_verl_trainer_main_ppo --> m_verl_workers_config_distillation
    m_verl_trainer_main_ppo --> m_verl_workers_engine_workers
    m_verl_trainer_main_ppo --> m_verl_workers_fsdp_workers
    m_verl_trainer_main_ppo --> m_verl_workers_megatron_workers
    m_verl_trainer_ppo_core_algos --> m_my_recipes_rllm_utils
    m_verl_trainer_ppo_core_algos --> m_verl_trainer_config_config
    m_verl_trainer_ppo_core_algos --> m_verl_trainer_ppo_rollout_corr_helper
    m_verl_trainer_ppo_core_algos --> m_verl_utils_import_utils
    m_verl_trainer_ppo_core_algos --> m_verl_utils_torch_functional
    m_verl_trainer_ppo_metric_utils --> m_verl_utils_import_utils
    m_verl_trainer_ppo_metric_utils --> m_verl_utils_torch_functional
    m_verl_trainer_ppo_prefix_grouper_utils --> m_verl_utils_torch_functional
    m_verl_trainer_ppo_ray_trainer --> m_my_recipes_rllm_utils
    m_verl_trainer_ppo_ray_trainer --> m_my_recipes_syntool_agent_loop
    m_verl_trainer_ppo_ray_trainer --> m_verl_experimental_dataset_sampler
    m_verl_trainer_ppo_ray_trainer --> m_verl_experimental_reward_loop_reward_loop
    m_verl_trainer_ppo_ray_trainer --> m_verl_protocol
    m_verl_trainer_ppo_ray_trainer --> m_verl_single_controller_ray_base
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_config_config
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_distillation_losses
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_main_ppo
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_ppo_core_algos
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_ppo_metric_utils
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_ppo_reward
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_ppo_rollout_corr_helper
    m_verl_trainer_ppo_ray_trainer --> m_verl_trainer_ppo_utils
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_config
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_dataset_rl_dataset
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_debug_metrics
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_fs
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_import_utils
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_py_functional
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_rollout_skip
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_seqlen_balancing
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_torch_functional
    m_verl_trainer_ppo_ray_trainer --> m_verl_utils_tracking
    m_verl_trainer_ppo_ray_trainer --> m_verl_workers_engine_workers
    m_verl_trainer_ppo_ray_trainer --> m_verl_workers_utils_losses
    m_verl_trainer_ppo_ray_trainer --> m_verl_workers_utils_padding
    m_verl_trainer_ppo_reward --> m_verl_experimental_reward_loop_reward_manager_base
    m_verl_trainer_ppo_reward --> m_verl_trainer_config_config
    m_verl_trainer_ppo_reward --> m_verl_utils_import_utils
    m_verl_trainer_ppo_reward --> m_verl_workers_config_reward
    m_verl_trainer_ppo_rollout_corr_helper --> m_verl_protocol
    m_verl_trainer_ppo_rollout_corr_helper --> m_verl_trainer_config_algorithm
    m_verl_trainer_ppo_rollout_corr_helper --> m_verl_utils_torch_functional
    m_verl_trainer_ppo_rollout_corr_helper --> m_verl_workers_config_actor
    m_verl_trainer_ppo_utils --> m_verl_checkpoint_engine_base
    m_verl_trainer_ppo_utils --> m_verl_trainer_ppo_core_algos
    m_verl_trainer_ppo_utils --> m_verl_workers_config_distillation
    m_verl_trainer_sft_trainer --> m_my_recipes_rllm_utils
    m_verl_trainer_sft_trainer --> m_verl_utils_config
    m_verl_trainer_sft_trainer --> m_verl_utils_dataset_dataset_utils
    m_verl_trainer_sft_trainer --> m_verl_utils_dataset_multiturn_sft_dataset
    m_verl_trainer_sft_trainer --> m_verl_utils_device
    m_verl_trainer_sft_trainer --> m_verl_utils_distributed
    m_verl_trainer_sft_trainer --> m_verl_utils_import_utils
    m_verl_trainer_sft_trainer --> m_verl_utils_memory_utils
    m_verl_trainer_sft_trainer --> m_verl_utils_tracking
    m_verl_trainer_sft_trainer --> m_verl_workers_engine_workers
    m_verl_trainer_sft_trainer --> m_verl_workers_utils_losses
    m_verl_trainer_sft_trainer_ray --> m_my_recipes_rllm_utils
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_config
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_dataset_dataset_utils
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_dataset_multiturn_sft_dataset
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_device
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_distributed
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_import_utils
    m_verl_trainer_sft_trainer_ray --> m_verl_utils_tracking
    m_verl_trainer_sft_trainer_ray --> m_verl_workers_engine_workers
    m_verl_trainer_sft_trainer_ray --> m_verl_workers_utils_losses
    m_verl_utils___init__ --> m_tests_special_e2e_envs_digit_completion_tokenizer
    m_verl_utils___init__ --> m_verl_trainer_config_config
    m_verl_utils___init__ --> m_verl_utils_groupwise
    m_verl_utils_activation_offload --> m_verl_utils_device
    m_verl_utils_activation_offload --> m_verl_utils_fsdp_utils
    m_verl_utils_attention_utils --> m_verl_utils_device
    m_verl_utils_attention_utils --> m_verl_utils_npu_flash_attn_utils
    m_verl_utils_chat_template --> m_verl_utils_tokenizer
    m_verl_utils_checkpoint___init__ --> m_verl_utils_checkpoint_checkpoint_handler
    m_verl_utils_checkpoint_checkpoint_handler --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_utils_checkpoint_checkpoint_handler --> m_verl_utils_fs
    m_verl_utils_checkpoint_checkpoint_handler --> m_verl_utils_hdfs_io
    m_verl_utils_checkpoint_checkpoint_handler --> m_verl_workers_config_engine
    m_verl_utils_checkpoint_checkpoint_manager --> m_verl_trainer_config_config
    m_verl_utils_checkpoint_checkpoint_manager --> m_verl_utils_device
    m_verl_utils_checkpoint_checkpoint_manager --> m_verl_utils_distributed
    m_verl_utils_checkpoint_fsdp_checkpoint_manager --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_utils_checkpoint_fsdp_checkpoint_manager --> m_verl_utils_device
    m_verl_utils_checkpoint_fsdp_checkpoint_manager --> m_verl_utils_distributed
    m_verl_utils_checkpoint_fsdp_checkpoint_manager --> m_verl_utils_fs
    m_verl_utils_checkpoint_fsdp_checkpoint_manager --> m_verl_utils_fsdp_utils
    m_verl_utils_checkpoint_fsdp_checkpoint_manager --> m_verl_utils_transformers_compat
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_my_recipes_rllm_utils
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_checkpoint_engine_base
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_models_weight_loader_registry
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_checkpoint_checkpoint_manager
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_device
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_distributed
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_fs
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_megatron_dist_checkpointing
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_megatron_utils
    m_verl_utils_checkpoint_megatron_checkpoint_manager --> m_verl_utils_qat_core
    m_verl_utils_config --> m_my_recipes_rllm_utils
    m_verl_utils_config --> m_verl_workers_rollout_vllm_rollout_utils
    m_verl_utils_dataset___init__ --> m_verl_utils_dataset_rl_dataset
    m_verl_utils_dataset___init__ --> m_verl_utils_dataset_rm_dataset
    m_verl_utils_dataset_multiturn_sft_dataset --> m_my_recipes_rllm_utils
    m_verl_utils_dataset_multiturn_sft_dataset --> m_verl_models_transformers_qwen2_vl
    m_verl_utils_dataset_multiturn_sft_dataset --> m_verl_utils_chat_template
    m_verl_utils_dataset_multiturn_sft_dataset --> m_verl_utils_dataset_dataset_utils
    m_verl_utils_dataset_multiturn_sft_dataset --> m_verl_utils_dataset_vision_utils
    m_verl_utils_dataset_multiturn_sft_dataset --> m_verl_utils_fs
    m_verl_utils_dataset_multiturn_sft_dataset --> m_verl_utils_py_functional
    m_verl_utils_dataset_rl_dataset --> m_verl_tools_utils_tool_registry
    m_verl_utils_dataset_rl_dataset --> m_verl_utils_dataset_vision_utils
    m_verl_utils_dataset_rl_dataset --> m_verl_utils_fs
    m_verl_utils_dataset_rl_dataset --> m_verl_utils_import_utils
    m_verl_utils_dataset_rl_dataset --> m_verl_utils_tokenizer
    m_verl_utils_dataset_rm_dataset --> m_my_recipes_rllm_utils
    m_verl_utils_dataset_rm_dataset --> m_verl_utils_distributed
    m_verl_utils_dataset_rm_dataset --> m_verl_utils_fs
    m_verl_utils_debug_metrics --> m_verl_protocol
    m_verl_utils_debug_performance --> m_verl_utils_profiler_performance
    m_verl_utils_debug_trajectory_tracker --> m_verl_utils_hdfs_io
    m_verl_utils_distributed --> m_my_recipes_rllm_utils
    m_verl_utils_distributed --> m_verl_utils_device
    m_verl_utils_distributed --> m_verl_utils_net_utils
    m_verl_utils_flops_counter --> m_verl_utils_device
    m_verl_utils_fp8_utils --> m_verl_utils_kernel_fp8_kernel
    m_verl_utils_fp8_utils --> m_verl_workers_rollout_utils
    m_verl_utils_fs --> m_verl_utils_hdfs_io
    m_verl_utils_fsdp_utils --> m_verl_third_party_torch_distributed_checkpoint_state_dict
    m_verl_utils_fsdp_utils --> m_verl_utils_device
    m_verl_utils_fsdp_utils --> m_verl_utils_distributed
    m_verl_utils_fsdp_utils --> m_verl_utils_model
    m_verl_utils_groupwise --> m_verl_utils_device
    m_verl_utils_import_utils --> m_verl_models_mcore_util
    m_verl_utils_kernel_kernels --> m_verl_utils_device
    m_verl_utils_kernel_kernels --> m_verl_utils_distributed
    m_verl_utils_kernel_linear_cross_entropy --> m_verl_utils_distributed
    m_verl_utils_logger___init__ --> m_verl_utils_logger_aggregate_logger
    m_verl_utils_megatron_dist_checkpointing --> m_verl_utils_qat_core
    m_verl_utils_megatron_memory --> m_verl_utils_device
    m_verl_utils_megatron_pipeline_parallel --> m_verl_utils_megatron_sequence_parallel
    m_verl_utils_megatron_pipeline_parallel --> m_verl_utils_qat_core
    m_verl_utils_megatron_router_replay_utils --> m_my_recipes_rllm_utils
    m_verl_utils_megatron_router_replay_utils --> m_verl_models_mcore_util
    m_verl_utils_megatron_router_replay_utils --> m_verl_utils_device
    m_verl_utils_megatron_router_replay_utils --> m_verl_utils_megatron_router_replay_patch
    m_verl_utils_megatron_router_replay_utils --> m_verl_utils_megatron_tensor_parallel
    m_verl_utils_megatron_router_replay_utils --> m_verl_utils_qat_core
    m_verl_utils_megatron_sequence_parallel --> m_verl_utils_qat_core
    m_verl_utils_megatron_tensor_parallel --> m_verl_utils_distributed
    m_verl_utils_megatron_tensor_parallel --> m_verl_utils_qat_core
    m_verl_utils_megatron_peft_utils --> m_verl_utils_megatron_utils
    m_verl_utils_megatron_utils --> m_my_recipes_rllm_utils
    m_verl_utils_megatron_utils --> m_verl_checkpoint_engine_base
    m_verl_utils_megatron_utils --> m_verl_models_mcore_bridge
    m_verl_utils_megatron_utils --> m_verl_models_mcore_config_converter
    m_verl_utils_megatron_utils --> m_verl_models_mcore_mbridge
    m_verl_utils_megatron_utils --> m_verl_models_mcore_mtp_patch
    m_verl_utils_megatron_utils --> m_verl_models_mcore_patch
    m_verl_utils_megatron_utils --> m_verl_trainer_config_config
    m_verl_utils_megatron_utils --> m_verl_utils_device
    m_verl_utils_megatron_utils --> m_verl_utils_distributed
    m_verl_utils_megatron_utils --> m_verl_utils_fs
    m_verl_utils_megatron_utils --> m_verl_utils_megatron_dist_checkpointing
    m_verl_utils_megatron_utils --> m_verl_utils_megatron_optimizer
    m_verl_utils_megatron_utils --> m_verl_utils_megatron_tensor_parallel
    m_verl_utils_megatron_utils --> m_verl_utils_megatron_peft_utils
    m_verl_utils_megatron_utils --> m_verl_utils_model
    m_verl_utils_megatron_utils --> m_verl_utils_qat_core
    m_verl_utils_megatron_utils --> m_verl_utils_torch_dtypes
    m_verl_utils_memory_utils --> m_verl_utils_device
    m_verl_utils_metric___init__ --> m_my_recipes_rllm_utils
    m_verl_utils_model --> m_verl_models_mcore_bridge
    m_verl_utils_model --> m_verl_models_mcore_config_converter
    m_verl_utils_model --> m_verl_models_mcore_loader
    m_verl_utils_model --> m_verl_models_mcore_saver
    m_verl_utils_model --> m_verl_models_registry
    m_verl_utils_model --> m_verl_models_weight_loader_registry
    m_verl_utils_model --> m_verl_utils_fs
    m_verl_utils_model --> m_verl_utils_import_utils
    m_verl_utils_model --> m_verl_utils_megatron_utils
    m_verl_utils_model --> m_verl_utils_qat_core
    m_verl_utils_model --> m_verl_utils_transformers_compat
    m_verl_utils_modelopt___init__ --> m_verl_utils_modelopt_megatron_qat_patch
    m_verl_utils_modelopt___init__ --> m_verl_utils_modelopt_qat_utils
    m_verl_utils_modelopt___init__ --> m_verl_utils_modelopt_qat_weight_exporter
    m_verl_utils_modelopt___init__ --> m_verl_utils_modelopt_quantize
    m_verl_utils_modelopt___init__ --> m_verl_utils_modelopt_vllm_modelopt_patch
    m_verl_utils_modelopt_megatron_qat_patch --> m_my_recipes_rllm_utils
    m_verl_utils_modelopt_megatron_qat_patch --> m_verl_experimental_vla_models_modules_mlp
    m_verl_utils_modelopt_megatron_qat_patch --> m_verl_utils_megatron_dist_checkpointing
    m_verl_utils_modelopt_megatron_qat_patch --> m_verl_utils_qat_core
    m_verl_utils_modelopt_qat_utils --> m_verl_utils_modelopt_megatron_qat_patch
    m_verl_utils_modelopt_qat_utils --> m_verl_utils_modelopt_qat_weight_exporter
    m_verl_utils_modelopt_qat_utils --> m_verl_utils_modelopt_quantize
    m_verl_utils_modelopt_qat_weight_exporter --> m_verl_utils_megatron_utils
    m_verl_utils_modelopt_qat_weight_exporter --> m_verl_utils_qat_core
    m_verl_utils_modelopt_quantize --> m_verl_trainer_config_config
    m_verl_utils_modelopt_vllm_modelopt_patch --> m_verl_utils_device
    m_verl_utils_profiler___init__ --> m_verl_trainer_config_config
    m_verl_utils_profiler___init__ --> m_verl_utils_debug_performance
    m_verl_utils_profiler___init__ --> m_verl_utils_device
    m_verl_utils_profiler___init__ --> m_verl_utils_import_utils
    m_verl_utils_profiler___init__ --> m_verl_utils_profiler_mstx_profile
    m_verl_utils_profiler___init__ --> m_verl_utils_profiler_nvtx_profile
    m_verl_utils_profiler___init__ --> m_verl_utils_profiler_profile
    m_verl_utils_profiler_config --> m_verl_base_config
    m_verl_utils_profiler_mstx_profile --> m_verl_trainer_config_config
    m_verl_utils_profiler_mstx_profile --> m_verl_utils_debug_performance
    m_verl_utils_profiler_mstx_profile --> m_verl_utils_profiler_profile
    m_verl_utils_profiler_nvtx_profile --> m_verl_trainer_config_config
    m_verl_utils_profiler_nvtx_profile --> m_verl_utils_debug_performance
    m_verl_utils_profiler_nvtx_profile --> m_verl_utils_profiler_profile
    m_verl_utils_profiler_performance --> m_verl_utils_device
    m_verl_utils_profiler_performance --> m_verl_utils_distributed
    m_verl_utils_profiler_profile --> m_verl_single_controller_base_decorator
    m_verl_utils_profiler_profile --> m_verl_trainer_config_config
    m_verl_utils_profiler_profile --> m_verl_utils_memory_utils
    m_verl_utils_profiler_profile --> m_verl_utils_profiler_mstx_profile
    m_verl_utils_profiler_profile --> m_verl_utils_profiler_nvtx_profile
    m_verl_utils_profiler_profile --> m_verl_utils_profiler_torch_profile
    m_verl_utils_profiler_torch_profile --> m_verl_trainer_config_config
    m_verl_utils_profiler_torch_profile --> m_verl_utils_profiler_profile
    m_verl_utils_qat___init__ --> m_verl_utils_qat_core
    m_verl_utils_qat___init__ --> m_verl_utils_qat_vllm_patch
    m_verl_utils_qat_core --> m_verl_base_config
    m_verl_utils_qat_core --> m_verl_utils_qat_linear
    m_verl_utils_qat_quantizer --> m_verl_utils_device
    m_verl_utils_qat_vllm_patch --> m_my_recipes_rllm_utils
    m_verl_utils_qat_vllm_patch --> m_verl_utils_device
    m_verl_utils_rendezvous_ray_backend --> m_verl_models_mcore_util
    m_verl_utils_reward_score___init__ --> m_verl_utils_import_utils
    m_verl_utils_reward_score_geo3k --> m_verl_utils_reward_score_prime_math_grader
    m_verl_utils_reward_score_math_batch --> m_my_recipes_rllm_rewards_math_reward
    m_verl_utils_reward_score_prime_code___init__ --> m_my_recipes_rllm_utils
    m_verl_utils_reward_score_prime_code_utils --> m_verl_utils_reward_score_prime_code_testing_util
    m_verl_utils_reward_score_prime_math___init__ --> m_verl_utils_py_functional
    m_verl_utils_reward_score_prime_math___init__ --> m_verl_utils_reward_score_prime_math_grader
    m_verl_utils_reward_score_prime_math_grader --> m_verl_utils_py_functional
    m_verl_utils_reward_score_sandbox_fusion___init__ --> m_my_recipes_rllm_utils
    m_verl_utils_rollout_skip --> m_verl_protocol
    m_verl_utils_rollout_skip --> m_verl_workers_config_rollout
    m_verl_utils_rollout_trace --> m_verl_utils_ray_utils
    m_verl_utils_seqlen_balancing --> m_my_recipes_rllm_utils
    m_verl_utils_seqlen_balancing --> m_verl_protocol
    m_verl_utils_seqlen_balancing --> m_verl_utils_device
    m_verl_utils_sglang_sglang_fp8_utils --> m_verl_utils_fp8_utils
    m_verl_utils_tokenizer --> m_verl_models_transformers_glm4v
    m_verl_utils_tokenizer --> m_verl_models_transformers_qwen2_vl
    m_verl_utils_tokenizer --> m_verl_models_transformers_qwen3_vl
    m_verl_utils_torch_functional --> m_verl_utils_attention_utils
    m_verl_utils_torch_functional --> m_verl_utils_device
    m_verl_utils_torch_functional --> m_verl_utils_distributed
    m_verl_utils_trtllm_trtllm_fp8_utils --> m_verl_utils_fp8_utils
    m_verl_utils_ulysses --> m_verl_utils_distributed
    m_verl_utils_vllm___init__ --> m_my_recipes_rllm_utils
    m_verl_utils_vllm___init__ --> m_verl_utils_vllm_npu_vllm_patch
    m_verl_utils_vllm_npu_vllm_patch --> m_my_recipes_rllm_utils
    m_verl_utils_vllm_npu_vllm_patch --> m_verl_utils_device
    m_verl_utils_vllm_patch --> m_verl_models_transformers_kimi_vl
    m_verl_utils_vllm_utils --> m_my_recipes_rllm_utils
    m_verl_utils_vllm_vllm_fp8_utils --> m_verl_utils_fp8_utils
    m_verl_utils_vllm_vllm_fp8_utils --> m_verl_utils_kernel_fp8_kernel
    m_verl_utils_vllm_vllm_fp8_utils --> m_verl_utils_qat_linear
    m_verl_utils_vllm_omni___init__ --> m_my_recipes_rllm_utils
    m_verl_utils_vllm_omni_utils --> m_my_recipes_rllm_utils
    m_verl_workers_actor___init__ --> m_verl_checkpoint_engine_base
    m_verl_workers_actor___init__ --> m_verl_workers_actor_dp_actor
    m_verl_workers_actor_dp_actor --> m_verl_trainer_config_config
    m_verl_workers_actor_dp_actor --> m_verl_trainer_ppo_core_algos
    m_verl_workers_actor_dp_actor --> m_verl_trainer_ppo_prefix_grouper_utils
    m_verl_workers_actor_dp_actor --> m_verl_trainer_ppo_rollout_corr_helper
    m_verl_workers_actor_dp_actor --> m_verl_utils_attention_utils
    m_verl_workers_actor_dp_actor --> m_verl_utils_dataset_vision_utils
    m_verl_workers_actor_dp_actor --> m_verl_utils_device
    m_verl_workers_actor_dp_actor --> m_verl_utils_fsdp_utils
    m_verl_workers_actor_dp_actor --> m_verl_utils_import_utils
    m_verl_workers_actor_dp_actor --> m_verl_utils_model
    m_verl_workers_actor_dp_actor --> m_verl_utils_py_functional
    m_verl_workers_actor_dp_actor --> m_verl_utils_seqlen_balancing
    m_verl_workers_actor_dp_actor --> m_verl_utils_torch_dtypes
    m_verl_workers_actor_dp_actor --> m_verl_utils_torch_functional
    m_verl_workers_actor_dp_actor --> m_verl_utils_ulysses
    m_verl_workers_actor_dp_actor --> m_verl_workers_config_actor
    m_verl_workers_actor_megatron_actor --> m_verl_models_mcore_model_forward_1f1b_overlap
    m_verl_workers_actor_megatron_actor --> m_verl_models_mcore_model_forward_fused
    m_verl_workers_actor_megatron_actor --> m_verl_models_mcore_mtp_patch
    m_verl_workers_actor_megatron_actor --> m_verl_trainer_config_config
    m_verl_workers_actor_megatron_actor --> m_verl_trainer_ppo_core_algos
    m_verl_workers_actor_megatron_actor --> m_verl_trainer_ppo_rollout_corr_helper
    m_verl_workers_actor_megatron_actor --> m_verl_utils_device
    m_verl_workers_actor_megatron_actor --> m_verl_utils_distributed
    m_verl_workers_actor_megatron_actor --> m_verl_utils_import_utils
    m_verl_workers_actor_megatron_actor --> m_verl_utils_megatron_optimizer
    m_verl_workers_actor_megatron_actor --> m_verl_utils_megatron_pipeline_parallel
    m_verl_workers_actor_megatron_actor --> m_verl_utils_megatron_router_replay_patch
    m_verl_workers_actor_megatron_actor --> m_verl_utils_megatron_router_replay_utils
    m_verl_workers_actor_megatron_actor --> m_verl_utils_megatron_tensor_parallel
    m_verl_workers_actor_megatron_actor --> m_verl_utils_megatron_utils
    m_verl_workers_actor_megatron_actor --> m_verl_utils_model
    m_verl_workers_actor_megatron_actor --> m_verl_utils_py_functional
    m_verl_workers_actor_megatron_actor --> m_verl_utils_qat_core
    m_verl_workers_actor_megatron_actor --> m_verl_utils_seqlen_balancing
    m_verl_workers_actor_megatron_actor --> m_verl_utils_torch_functional
    m_verl_workers_actor_megatron_actor --> m_verl_workers_config_actor
    m_verl_workers_config___init__ --> m_my_recipes_syntool_reward
    m_verl_workers_config___init__ --> m_verl_models_mcore_qwen2_5_vl_model
    m_verl_workers_config___init__ --> m_verl_utils_megatron_optimizer
    m_verl_workers_config___init__ --> m_verl_workers_config_actor
    m_verl_workers_config___init__ --> m_verl_workers_config_critic
    m_verl_workers_config___init__ --> m_verl_workers_config_distillation
    m_verl_workers_config___init__ --> m_verl_workers_config_engine
    m_verl_workers_config___init__ --> m_verl_workers_config_rollout
    m_verl_workers_config_actor --> m_verl_base_config
    m_verl_workers_config_actor --> m_verl_models_mcore_qwen2_5_vl_model
    m_verl_workers_config_actor --> m_verl_trainer_config_config
    m_verl_workers_config_actor --> m_verl_utils_megatron_optimizer
    m_verl_workers_config_actor --> m_verl_utils_profiler_config
    m_verl_workers_config_actor --> m_verl_workers_config_engine
    m_verl_workers_config_critic --> m_verl_base_config
    m_verl_workers_config_critic --> m_verl_models_mcore_qwen2_5_vl_model
    m_verl_workers_config_critic --> m_verl_trainer_config_config
    m_verl_workers_config_critic --> m_verl_utils_megatron_optimizer
    m_verl_workers_config_critic --> m_verl_workers_config_engine
    m_verl_workers_config_distillation --> m_verl_base_config
    m_verl_workers_config_distillation --> m_verl_trainer_distillation_losses
    m_verl_workers_config_distillation --> m_verl_workers_config_rollout
    m_verl_workers_config_engine --> m_verl_base_config
    m_verl_workers_config_engine --> m_verl_models_mcore_qwen2_5_vl_model
    m_verl_workers_config_engine --> m_verl_trainer_config_config
    m_verl_workers_config_engine --> m_verl_utils_megatron_optimizer
    m_verl_workers_config_megatron_peft --> m_verl_models_mcore_bridge
    m_verl_workers_config_megatron_peft --> m_verl_utils_torch_dtypes
    m_verl_workers_config_model --> m_my_recipes_rllm_utils
    m_verl_workers_config_model --> m_verl_base_config
    m_verl_workers_config_model --> m_verl_utils_fs
    m_verl_workers_config_model --> m_verl_utils_import_utils
    m_verl_workers_config_model --> m_verl_utils_model
    m_verl_workers_config_optimizer --> m_verl_base_config
    m_verl_workers_config_reward --> m_verl_base_config
    m_verl_workers_config_reward --> m_verl_experimental_reward_loop_reward_manager_registry
    m_verl_workers_config_reward --> m_verl_trainer_config_config
    m_verl_workers_config_reward --> m_verl_workers_config_rollout
    m_verl_workers_config_rollout --> m_verl_base_config
    m_verl_workers_config_rollout --> m_verl_workers_config_model
    m_verl_workers_critic___init__ --> m_verl_checkpoint_engine_base
    m_verl_workers_critic___init__ --> m_verl_workers_critic_dp_critic
    m_verl_workers_critic_dp_critic --> m_verl_utils_attention_utils
    m_verl_workers_critic_dp_critic --> m_verl_utils_device
    m_verl_workers_critic_dp_critic --> m_verl_utils_distributed
    m_verl_workers_critic_dp_critic --> m_verl_utils_fsdp_utils
    m_verl_workers_critic_dp_critic --> m_verl_utils_model
    m_verl_workers_critic_dp_critic --> m_verl_utils_py_functional
    m_verl_workers_critic_dp_critic --> m_verl_utils_seqlen_balancing
    m_verl_workers_critic_dp_critic --> m_verl_utils_torch_functional
    m_verl_workers_critic_dp_critic --> m_verl_utils_ulysses
    m_verl_workers_critic_dp_critic --> m_verl_workers_config_critic
    m_verl_workers_critic_megatron_critic --> m_verl_utils_device
    m_verl_workers_critic_megatron_critic --> m_verl_utils_distributed
    m_verl_workers_critic_megatron_critic --> m_verl_utils_megatron_optimizer
    m_verl_workers_critic_megatron_critic --> m_verl_utils_megatron_pipeline_parallel
    m_verl_workers_critic_megatron_critic --> m_verl_utils_py_functional
    m_verl_workers_critic_megatron_critic --> m_verl_utils_qat_core
    m_verl_workers_critic_megatron_critic --> m_verl_utils_seqlen_balancing
    m_verl_workers_critic_megatron_critic --> m_verl_utils_torch_functional
    m_verl_workers_critic_megatron_critic --> m_verl_workers_config_critic
    m_verl_workers_engine___init__ --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_automodel___init__ --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_automodel_transformer_impl --> m_my_recipes_rllm_utils
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_experimental_vla_models_openvla_oft_constants
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_models_mcore_loader
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_trainer_config_config
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_utils_dataset_dataset_utils
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_utils_device
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_utils_distributed
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_utils_model
    m_verl_workers_engine_automodel_transformer_impl --> m_verl_utils_torch_functional
    m_verl_workers_engine_automodel_utils --> m_my_recipes_rllm_utils
    m_verl_workers_engine_automodel_utils --> m_verl_trainer_config_config
    m_verl_workers_engine_automodel_utils --> m_verl_utils_device
    m_verl_workers_engine_automodel_utils --> m_verl_utils_distributed
    m_verl_workers_engine_automodel_utils --> m_verl_utils_torch_dtypes
    m_verl_workers_engine_base --> m_verl_utils_device
    m_verl_workers_engine_base --> m_verl_utils_tensordict_utils
    m_verl_workers_engine_fsdp___init__ --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_fsdp_transformer_impl --> m_my_recipes_rllm_utils
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_models_transformers_monkey_patch
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_trainer_config_config
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_activation_offload
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_checkpoint_fsdp_checkpoint_manager
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_dataset_dataset_utils
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_device
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_distributed
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_fs
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_fsdp_utils
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_model
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_py_functional
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_qat_core
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_qat_quantizer
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_torch_dtypes
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_torch_functional
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_utils_ulysses
    m_verl_workers_engine_fsdp_transformer_impl --> m_verl_workers_config_optimizer
    m_verl_workers_engine_fsdp_utils --> m_verl_models_transformers_npu_patch
    m_verl_workers_engine_fsdp_utils --> m_verl_utils_device
    m_verl_workers_engine_megatron___init__ --> m_verl_utils_device
    m_verl_workers_engine_megatron___init__ --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_megatron_transformer_impl --> m_my_recipes_rllm_utils
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_models_mcore_bridge
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_models_mcore_mbridge
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_models_mcore_model_forward_fused
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_models_mcore_patch
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_trainer_config_config
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_checkpoint_megatron_checkpoint_manager
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_dataset_dataset_utils
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_device
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_distributed
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_optimizer
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_pipeline_parallel
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_router_replay_patch
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_router_replay_utils
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_tensor_parallel
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_peft_utils
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_megatron_utils
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_model
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_qat_core
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_seqlen_balancing
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_torch_dtypes
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_utils_torch_functional
    m_verl_workers_engine_megatron_transformer_impl --> m_verl_workers_config_megatron_peft
    m_verl_workers_engine_megatron_utils --> m_verl_utils_device
    m_verl_workers_engine_megatron_utils --> m_verl_utils_qat_core
    m_verl_workers_engine_mindspeed___init__ --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_mindspeed_transformer_impl --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_mindspeed_transformer_impl --> m_verl_trainer_config_config
    m_verl_workers_engine_torchtitan___init__ --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_torchtitan_transformer_impl --> m_my_recipes_rllm_utils
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_third_party_torch_distributed_checkpoint_state_dict
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_trainer_config_config
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_dataset_dataset_utils
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_device
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_distributed
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_fsdp_utils
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_megatron_optimizer
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_model
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_utils_torch_functional
    m_verl_workers_engine_torchtitan_transformer_impl --> m_verl_workers_engine_torchtitan_utils
    m_verl_workers_engine_torchtitan_utils --> m_verl_models_mcore_qwen2_5_vl_attention
    m_verl_workers_engine_torchtitan_utils --> m_verl_utils_distributed
    m_verl_workers_engine_utils --> m_my_recipes_rllm_utils
    m_verl_workers_engine_utils --> m_verl_utils_dataset_dataset_utils
    m_verl_workers_engine_utils --> m_verl_utils_device
    m_verl_workers_engine_utils --> m_verl_utils_py_functional
    m_verl_workers_engine_utils --> m_verl_utils_seqlen_balancing
    m_verl_workers_engine_veomni___init__ --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_veomni_transformer_impl --> m_my_recipes_rllm_utils
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_trainer_config_config
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_checkpoint_fsdp_checkpoint_manager
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_device
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_distributed
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_fsdp_utils
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_model
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_torch_functional
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_utils_ulysses
    m_verl_workers_engine_veomni_transformer_impl --> m_verl_workers_engine_automodel_transformer_impl
    m_verl_workers_engine_veomni_utils --> m_verl_utils_device
    m_verl_workers_engine_workers --> m_my_recipes_rllm_utils
    m_verl_workers_engine_workers --> m_verl_checkpoint_engine_base
    m_verl_workers_engine_workers --> m_verl_single_controller_base_decorator
    m_verl_workers_engine_workers --> m_verl_trainer_config_config
    m_verl_workers_engine_workers --> m_verl_utils_config
    m_verl_workers_engine_workers --> m_verl_utils_device
    m_verl_workers_engine_workers --> m_verl_utils_distributed
    m_verl_workers_engine_workers --> m_verl_utils_flops_counter
    m_verl_workers_engine_workers --> m_verl_utils_memory_utils
    m_verl_workers_engine_workers --> m_verl_utils_metric_utils
    m_verl_workers_engine_workers --> m_verl_utils_py_functional
    m_verl_workers_engine_workers --> m_verl_utils_tensordict_utils
    m_verl_workers_engine_workers --> m_verl_utils_torch_functional
    m_verl_workers_engine_workers --> m_verl_workers_config_distillation
    m_verl_workers_engine_workers --> m_verl_workers_config_engine
    m_verl_workers_engine_workers --> m_verl_workers_rollout_base
    m_verl_workers_engine_workers --> m_verl_workers_utils_losses
    m_verl_workers_fsdp_workers --> m_my_recipes_rllm_utils
    m_verl_workers_fsdp_workers --> m_verl_checkpoint_engine_base
    m_verl_workers_fsdp_workers --> m_verl_models_transformers_monkey_patch
    m_verl_workers_fsdp_workers --> m_verl_single_controller_base_decorator
    m_verl_workers_fsdp_workers --> m_verl_trainer_config_config
    m_verl_workers_fsdp_workers --> m_verl_utils_activation_offload
    m_verl_workers_fsdp_workers --> m_verl_utils_checkpoint_fsdp_checkpoint_manager
    m_verl_workers_fsdp_workers --> m_verl_utils_config
    m_verl_workers_fsdp_workers --> m_verl_utils_device
    m_verl_workers_fsdp_workers --> m_verl_utils_distributed
    m_verl_workers_fsdp_workers --> m_verl_utils_flops_counter
    m_verl_workers_fsdp_workers --> m_verl_utils_fs
    m_verl_workers_fsdp_workers --> m_verl_utils_fsdp_utils
    m_verl_workers_fsdp_workers --> m_verl_utils_import_utils
    m_verl_workers_fsdp_workers --> m_verl_utils_memory_utils
    m_verl_workers_fsdp_workers --> m_verl_utils_model
    m_verl_workers_fsdp_workers --> m_verl_utils_profiler_performance
    m_verl_workers_fsdp_workers --> m_verl_utils_py_functional
    m_verl_workers_fsdp_workers --> m_verl_utils_qat_quantizer
    m_verl_workers_fsdp_workers --> m_verl_utils_ray_utils
    m_verl_workers_fsdp_workers --> m_verl_utils_torch_dtypes
    m_verl_workers_fsdp_workers --> m_verl_utils_torch_functional
    m_verl_workers_fsdp_workers --> m_verl_utils_transformers_compat
    m_verl_workers_fsdp_workers --> m_verl_workers_config_actor
    m_verl_workers_fsdp_workers --> m_verl_workers_config_critic
    m_verl_workers_fsdp_workers --> m_verl_workers_config_optimizer
    m_verl_workers_fsdp_workers --> m_verl_workers_config_rollout
    m_verl_workers_fsdp_workers --> m_verl_workers_engine_fsdp_utils
    m_verl_workers_fsdp_workers --> m_verl_workers_sharding_manager_fsdp_ulysses
    m_verl_workers_megatron_workers --> m_my_recipes_rllm_utils
    m_verl_workers_megatron_workers --> m_verl_checkpoint_engine_base
    m_verl_workers_megatron_workers --> m_verl_models_mcore_bridge
    m_verl_workers_megatron_workers --> m_verl_models_mcore_config_converter
    m_verl_workers_megatron_workers --> m_verl_models_mcore_mbridge
    m_verl_workers_megatron_workers --> m_verl_single_controller_base_decorator
    m_verl_workers_megatron_workers --> m_verl_trainer_config_config
    m_verl_workers_megatron_workers --> m_verl_utils_checkpoint_megatron_checkpoint_manager
    m_verl_workers_megatron_workers --> m_verl_utils_config
    m_verl_workers_megatron_workers --> m_verl_utils_device
    m_verl_workers_megatron_workers --> m_verl_utils_distributed
    m_verl_workers_megatron_workers --> m_verl_utils_flops_counter
    m_verl_workers_megatron_workers --> m_verl_utils_fs
    m_verl_workers_megatron_workers --> m_verl_utils_import_utils
    m_verl_workers_megatron_workers --> m_verl_utils_megatron_optimizer
    m_verl_workers_megatron_workers --> m_verl_utils_megatron_router_replay_patch
    m_verl_workers_megatron_workers --> m_verl_utils_megatron_peft_utils
    m_verl_workers_megatron_workers --> m_verl_utils_megatron_utils
    m_verl_workers_megatron_workers --> m_verl_utils_memory_utils
    m_verl_workers_megatron_workers --> m_verl_utils_model
    m_verl_workers_megatron_workers --> m_verl_utils_profiler_performance
    m_verl_workers_megatron_workers --> m_verl_utils_qat_core
    m_verl_workers_megatron_workers --> m_verl_utils_ray_utils
    m_verl_workers_megatron_workers --> m_verl_utils_torch_dtypes
    m_verl_workers_megatron_workers --> m_verl_utils_torch_functional
    m_verl_workers_megatron_workers --> m_verl_workers_actor_megatron_actor
    m_verl_workers_megatron_workers --> m_verl_workers_config_megatron_peft
    m_verl_workers_megatron_workers --> m_verl_workers_config_rollout
    m_verl_workers_megatron_workers --> m_verl_workers_critic_megatron_critic
    m_verl_workers_megatron_workers --> m_verl_workers_engine_mindspeed_transformer_impl
    m_verl_workers_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_dapo
    m_verl_workers_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_limited
    m_verl_workers_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_naive
    m_verl_workers_reward_manager___init__ --> m_verl_experimental_reward_loop_reward_manager_registry
    m_verl_workers_reward_manager___init__ --> m_verl_workers_reward_manager_batch
    m_verl_workers_reward_manager___init__ --> m_verl_workers_reward_manager_prime
    m_verl_workers_reward_manager_abstract --> m_verl_protocol
    m_verl_workers_reward_manager_batch --> m_verl_workers_reward_manager_abstract
    m_verl_workers_reward_manager_dapo --> m_verl_workers_reward_manager_abstract
    m_verl_workers_reward_manager_naive --> m_verl_workers_reward_manager_abstract
    m_verl_workers_reward_manager_prime --> m_verl_utils_ray_utils
    m_verl_workers_reward_manager_prime --> m_verl_workers_reward_manager_abstract
    m_verl_workers_reward_manager_registry --> m_verl_workers_reward_manager_abstract
    m_verl_workers_rollout___init__ --> m_verl_checkpoint_engine_base
    m_verl_workers_rollout___init__ --> m_verl_experimental_reward_loop_reward_manager_naive
    m_verl_workers_rollout___init__ --> m_verl_workers_rollout_hf_rollout
    m_verl_workers_rollout___init__ --> m_verl_workers_rollout_replica
    m_verl_workers_rollout_base --> m_verl_trainer_config_config
    m_verl_workers_rollout_base --> m_verl_utils_config
    m_verl_workers_rollout_hf_rollout --> m_verl_checkpoint_engine_base
    m_verl_workers_rollout_hf_rollout --> m_verl_utils_device
    m_verl_workers_rollout_hf_rollout --> m_verl_utils_distributed
    m_verl_workers_rollout_hf_rollout --> m_verl_utils_torch_functional
    m_verl_workers_rollout_naive___init__ --> m_verl_workers_rollout_naive_naive_rollout
    m_verl_workers_rollout_naive_naive_rollout --> m_verl_checkpoint_engine_base
    m_verl_workers_rollout_naive_naive_rollout --> m_verl_utils_torch_functional
    m_verl_workers_rollout_replica --> m_verl_checkpoint_engine_base
    m_verl_workers_rollout_replica --> m_verl_trainer_config_config
    m_verl_workers_rollout_replica --> m_verl_utils_config
    m_verl_workers_rollout_replica --> m_verl_utils_device
    m_verl_workers_rollout_replica --> m_verl_workers_config_actor
    m_verl_workers_rollout_replica --> m_verl_workers_rollout_sglang_rollout_async_sglang_server
    m_verl_workers_rollout_replica --> m_verl_workers_rollout_trtllm_rollout_trtllm_async_server
    m_verl_workers_rollout_replica --> m_verl_workers_rollout_vllm_rollout_vllm_async_server
    m_verl_workers_rollout_replica --> m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server
    m_verl_workers_rollout_schemas --> m_verl_models_transformers_qwen2_vl
    m_verl_workers_rollout_schemas --> m_verl_tools_schemas
    m_verl_workers_rollout_schemas --> m_verl_utils_model
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_trainer_config_config
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_utils_config
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_utils_device
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_utils_net_utils
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_workers_config_actor
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_workers_config_engine
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_workers_rollout_replica
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_workers_rollout_sglang_rollout_sglang_rollout
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_workers_rollout_sglang_rollout_utils
    m_verl_workers_rollout_sglang_rollout_async_sglang_server --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_sglang_rollout_http_server_engine --> m_my_recipes_rllm_utils
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_my_recipes_rllm_utils
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_trainer_config_config
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_utils_net_utils
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_utils_sglang_sglang_fp8_utils
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_workers_config_engine
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_workers_rollout_base
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_workers_rollout_sglang_rollout_http_server_engine
    m_verl_workers_rollout_sglang_rollout_sglang_rollout --> m_verl_workers_rollout_sglang_rollout_utils
    m_verl_workers_rollout_sglang_rollout_utils --> m_verl_utils_device
    m_verl_workers_rollout_sglang_rollout_utils --> m_verl_utils_distributed
    m_verl_workers_rollout_sglang_rollout_utils --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_models_mcore_util
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_trainer_config_config
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_utils_config
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_utils_net_utils
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_workers_config_actor
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_workers_rollout_replica
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_workers_rollout_trtllm_rollout_trtllm_rollout
    m_verl_workers_rollout_trtllm_rollout_trtllm_async_server --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_trainer_config_config
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_utils_device
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_utils_distributed
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_utils_net_utils
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_utils_trtllm_trtllm_fp8_utils
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_workers_rollout_base
    m_verl_workers_rollout_trtllm_rollout_trtllm_rollout --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_trtllm_rollout_trtllm_worker_extension --> m_my_recipes_rllm_utils
    m_verl_workers_rollout_vllm_rollout___init__ --> m_verl_workers_rollout_vllm_rollout_vllm_rollout
    m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer --> m_verl_utils_device
    m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_vllm_rollout_utils --> m_my_recipes_rllm_utils
    m_verl_workers_rollout_vllm_rollout_utils --> m_verl_utils_device
    m_verl_workers_rollout_vllm_rollout_utils --> m_verl_utils_modelopt_vllm_modelopt_patch
    m_verl_workers_rollout_vllm_rollout_utils --> m_verl_utils_vllm_patch
    m_verl_workers_rollout_vllm_rollout_utils --> m_verl_utils_vllm_vllm_fp8_utils
    m_verl_workers_rollout_vllm_rollout_utils --> m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_my_recipes_rllm_utils
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_trainer_config_config
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_utils_config
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_utils_device
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_utils_net_utils
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_utils_tokenizer
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_utils_vllm_npu_vllm_patch
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_utils_vllm_vllm_fp8_utils
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_workers_config_actor
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_workers_config_engine
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_workers_rollout_replica
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_vllm_rollout_vllm_async_server --> m_verl_workers_rollout_vllm_rollout_utils
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_trainer_config_config
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_utils_config
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_utils_tokenizer
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_workers_rollout_replica
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_workers_rollout_utils
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_workers_rollout_vllm_rollout_utils
    m_verl_workers_rollout_vllm_rollout_vllm_omni_async_server --> m_verl_workers_rollout_vllm_rollout_vllm_async_server
    m_verl_workers_rollout_vllm_rollout_vllm_rollout --> m_verl_trainer_config_config
    m_verl_workers_rollout_vllm_rollout_vllm_rollout --> m_verl_utils_device
    m_verl_workers_rollout_vllm_rollout_vllm_rollout --> m_verl_workers_rollout_base
    m_verl_workers_rollout_vllm_rollout_vllm_rollout --> m_verl_workers_rollout_vllm_rollout_bucketed_weight_transfer
    m_verl_workers_rollout_vllm_rollout_vllm_rollout --> m_verl_workers_rollout_vllm_rollout_utils
    m_verl_workers_sharding_manager_fsdp_ulysses --> m_verl_checkpoint_engine_base
    m_verl_workers_sharding_manager_fsdp_ulysses --> m_verl_protocol
    m_verl_workers_sharding_manager_fsdp_ulysses --> m_verl_utils_ulysses
    m_verl_workers_utils_losses --> m_my_recipes_rllm_utils
    m_verl_workers_utils_losses --> m_verl_trainer_config_config
    m_verl_workers_utils_losses --> m_verl_trainer_ppo_core_algos
    m_verl_workers_utils_losses --> m_verl_utils_dataset_dataset_utils
    m_verl_workers_utils_losses --> m_verl_utils_torch_functional
    m_verl_workers_utils_losses --> m_verl_workers_utils_padding
    m_verl_workers_utils_padding --> m_my_recipes_rllm_utils
    m_verl_workers_utils_padding --> m_verl_utils_attention_utils
```

## Module Index

| Module | File | Internal deps | External deps (sample) |
|---|---|---:|---|
| `docs/_static/js/resizable-sidebar` | `docs/_static/js/resizable-sidebar.js` | 0 | - |
| `docs/_static/js/runllm-widget` | `docs/_static/js/runllm-widget.js` | 0 | - |
| `docs/conf` | `docs/conf.py` | 0 | - |
| `examples/data_preprocess/aime2024_multiturn_w_tool` | `examples/data_preprocess/aime2024_multiturn_w_tool.py` | 1 | argparse, datasets, os |
| `examples/data_preprocess/dapo_multiturn_w_tool` | `examples/data_preprocess/dapo_multiturn_w_tool.py` | 1 | argparse, datasets, os |
| `examples/data_preprocess/full_hh_rlhf` | `examples/data_preprocess/full_hh_rlhf.py` | 1 | argparse, datasets, os, pandas, tqdm.auto |
| `examples/data_preprocess/geo3k` | `examples/data_preprocess/geo3k.py` | 1 | argparse, datasets, os |
| `examples/data_preprocess/geo3k_multiturn_w_tool` | `examples/data_preprocess/geo3k_multiturn_w_tool.py` | 1 | argparse, datasets, os |
| `examples/data_preprocess/gsm8k` | `examples/data_preprocess/gsm8k.py` | 1 | argparse, datasets, os, re |
| `examples/data_preprocess/gsm8k_multiturn_sft` | `examples/data_preprocess/gsm8k_multiturn_sft.py` | 1 | argparse, datasets, os, re |
| `examples/data_preprocess/gsm8k_multiturn_w_interaction` | `examples/data_preprocess/gsm8k_multiturn_w_interaction.py` | 1 | argparse, datasets, os, re |
| `examples/data_preprocess/gsm8k_multiturn_w_tool` | `examples/data_preprocess/gsm8k_multiturn_w_tool.py` | 1 | argparse, datasets, os, re |
| `examples/data_preprocess/gsm8k_tool_agent_loop` | `examples/data_preprocess/gsm8k_tool_agent_loop.py` | 1 | argparse, datasets, os, re |
| `examples/data_preprocess/hellaswag` | `examples/data_preprocess/hellaswag.py` | 1 | argparse, datasets, os, re |
| `examples/data_preprocess/math_dataset` | `examples/data_preprocess/math_dataset.py` | 2 | argparse, datasets, json, os |
| `examples/data_preprocess/multiturn` | `examples/data_preprocess/multiturn.py` | 1 | argparse, os, pandas |
| `examples/data_preprocess/pokemon` | `examples/data_preprocess/pokemon.py` | 1 | argparse, datasets, os |
| `examples/data_preprocess/preprocess_search_r1_dataset` | `examples/data_preprocess/preprocess_search_r1_dataset.py` | 2 | argparse, huggingface_hub, logging, os, pandas |
| `examples/fapo_trainer/prepare_data` | `examples/fapo_trainer/prepare_data.py` | 1 | argparse, datasets, functools, os, random |
| `examples/fapo_trainer/reward_fn` | `examples/fapo_trainer/reward_fn.py` | 2 | aiohttp, asyncio, logging, os, transformers |
| `examples/flowgrpo_trainer/reward_fn` | `examples/flowgrpo_trainer/reward_fn.py` | 2 | Levenshtein, PIL, aiohttp, json, numpy |
| `examples/sglang_multiturn/gsm8k_toolcall_shaping/gsm8k_toolcall_shaping` | `examples/sglang_multiturn/gsm8k_toolcall_shaping/gsm8k_toolcall_shaping.py` | 1 | __future__, typing |
| `examples/sglang_multiturn/search_r1_like/local_dense_retriever/download` | `examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py` | 0 | argparse, huggingface_hub |
| `examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server` | `examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py` | 0 | argparse, datasets, faiss, fastapi, json |
| `examples/split_placement/main_ppo_split` | `examples/split_placement/main_ppo_split.py` | 7 | hydra, omegaconf, pprint, ray, torch |
| `examples/split_placement/split_monkey_patch` | `examples/split_placement/split_monkey_patch.py` | 3 | copy, numpy, omegaconf, pprint, torch |
| `examples/tutorial/agent_loop_get_started/sandbox` | `examples/tutorial/agent_loop_get_started/sandbox.py` | 2 | aiohttp, re |
| `examples/vllm_omni/pipeline_qwenimage` | `examples/vllm_omni/pipeline_qwenimage.py` | 2 | diffusers.models.autoencoders.autoencoder_kl_qwenimage, os, torch, transformers, typing |
| `examples/vllm_omni/scheduling_flow_match_sde_discrete` | `examples/vllm_omni/scheduling_flow_match_sde_discrete.py` | 1 | dataclasses, diffusers, diffusers.utils.torch_utils, math, torch |
| `my_recipes/__init__` | `my_recipes/__init__.py` | 0 | - |
| `my_recipes/genrm_remote/reward_function` | `my_recipes/genrm_remote/reward_function.py` | 1 | concurrent.futures, requests, time, verl.utils.reward_score |
| `my_recipes/genrm_remote/reward_function_genrm` | `my_recipes/genrm_remote/reward_function_genrm.py` | 1 | concurrent.futures, json, re, requests, time |
| `my_recipes/gym/prepare_qwen_gym_parquet` | `my_recipes/gym/prepare_qwen_gym_parquet.py` | 0 | __future__, argparse, datasets, json, pathlib |
| `my_recipes/insturct_following/__init__` | `my_recipes/insturct_following/__init__.py` | 0 | insturct_following |
| `my_recipes/insturct_following/evaluation_main` | `my_recipes/insturct_following/evaluation_main.py` | 1 | absl, collections, dataclasses, json, os |
| `my_recipes/insturct_following/instructions` | `my_recipes/insturct_following/instructions.py` | 1 | absl, collections, json, langdetect, random |
| `my_recipes/insturct_following/instructions_registry` | `my_recipes/insturct_following/instructions_registry.py` | 1 | - |
| `my_recipes/insturct_following/instructions_util` | `my_recipes/insturct_following/instructions_util.py` | 0 | functools, immutabledict, nltk, random, re |
| `my_recipes/insturct_following/reward_function` | `my_recipes/insturct_following/reward_function.py` | 1 | json, os, typing |
| `my_recipes/logic_rl/reward_fn` | `my_recipes/logic_rl/reward_fn.py` | 0 | re, typing |
| `my_recipes/rllm/__init__` | `my_recipes/rllm/__init__.py` | 0 | - |
| `my_recipes/rllm/globals` | `my_recipes/rllm/globals.py` | 0 | - |
| `my_recipes/rllm/rewards/__init__` | `my_recipes/rllm/rewards/__init__.py` | 1 | - |
| `my_recipes/rllm/rewards/code_reward` | `my_recipes/rllm/rewards/code_reward.py` | 10 | ast, itertools, json, multiprocessing, os |
| `my_recipes/rllm/rewards/code_utils/codeforces` | `my_recipes/rllm/rewards/code_utils/codeforces.py` | 1 | builtins, datetime, enum, faulthandler, inspect |
| `my_recipes/rllm/rewards/code_utils/firejail_exec` | `my_recipes/rllm/rewards/code_utils/firejail_exec.py` | 1 | os, subprocess, tempfile |
| `my_recipes/rllm/rewards/code_utils/humanevalplus` | `my_recipes/rllm/rewards/code_utils/humanevalplus.py` | 1 | ast, builtins, faulthandler, os, platform |
| `my_recipes/rllm/rewards/code_utils/kodcode` | `my_recipes/rllm/rewards/code_utils/kodcode.py` | 1 | builtins, faulthandler, os, platform, pytest |
| `my_recipes/rllm/rewards/code_utils/livecodebench` | `my_recipes/rllm/rewards/code_utils/livecodebench.py` | 1 | ast, builtins, datetime, decimal, enum |
| `my_recipes/rllm/rewards/code_utils/pyext2` | `my_recipes/rllm/rewards/code_utils/pyext2.py` | 0 | IPython, sys |
| `my_recipes/rllm/rewards/code_utils/taco` | `my_recipes/rllm/rewards/code_utils/taco.py` | 2 | ast, builtins, datetime, enum, faulthandler |
| `my_recipes/rllm/rewards/code_utils/utils` | `my_recipes/rllm/rewards/code_utils/utils.py` | 0 | bisect, collections, copy, datetime, functools |
| `my_recipes/rllm/rewards/math_reward` | `my_recipes/rllm/rewards/math_reward.py` | 3 | json, os, rewards, typing |
| `my_recipes/rllm/rewards/math_utils/__init__` | `my_recipes/rllm/rewards/math_utils/__init__.py` | 1 | os |
| `my_recipes/rllm/rewards/math_utils/utils` | `my_recipes/rllm/rewards/math_utils/utils.py` | 0 | pylatexenc, re, sympy, sympy.parsing, typing |
| `my_recipes/rllm/rewards/reward_types` | `my_recipes/rllm/rewards/reward_types.py` | 0 | dataclasses, enum |
| `my_recipes/rllm/rewards/rl_reward` | `my_recipes/rllm/rewards/rl_reward.py` | 3 | json, os, typing |
| `my_recipes/rllm/system_prompts` | `my_recipes/rllm/system_prompts.py` | 0 | - |
| `my_recipes/rllm/tools/__init__` | `my_recipes/rllm/tools/__init__.py` | 0 | os, tools.code_tools, tools.math_tools, tools.web_tools |
| `my_recipes/rllm/tools/code_tools/__init__` | `my_recipes/rllm/tools/code_tools/__init__.py` | 3 | os |
| `my_recipes/rllm/tools/code_tools/code_tool` | `my_recipes/rllm/tools/code_tools/code_tool.py` | 1 | abc, dataclasses, os, typing |
| `my_recipes/rllm/tools/code_tools/e2b_tool` | `my_recipes/rllm/tools/code_tools/e2b_tool.py` | 1 | asyncio, e2b_code_interpreter, os, pprint, typing |
| `my_recipes/rllm/tools/code_tools/lcb_tool` | `my_recipes/rllm/tools/code_tools/lcb_tool.py` | 2 | ast, faulthandler, multiprocessing, os, queue |
| `my_recipes/rllm/tools/code_tools/local_tool` | `my_recipes/rllm/tools/code_tools/local_tool.py` | 1 | asyncio, concurrent.futures, contextlib, io, json |
| `my_recipes/rllm/tools/code_tools/together_tool` | `my_recipes/rllm/tools/code_tools/together_tool.py` | 1 | io, os, sys, together, typing |
| `my_recipes/rllm/tools/code_tools/utils` | `my_recipes/rllm/tools/code_tools/utils.py` | 1 | contextlib, io, os, sys, traceback |
| `my_recipes/rllm/tools/example_tool` | `my_recipes/rllm/tools/example_tool.py` | 1 | asyncio, os, time, typing |
| `my_recipes/rllm/tools/math_tools/__init__` | `my_recipes/rllm/tools/math_tools/__init__.py` | 1 | os |
| `my_recipes/rllm/tools/math_tools/calculator` | `my_recipes/rllm/tools/math_tools/calculator.py` | 1 | asyncio, os |
| `my_recipes/rllm/tools/multi_tool` | `my_recipes/rllm/tools/multi_tool.py` | 1 | asyncio, os, tools, typing |
| `my_recipes/rllm/tools/tool_base` | `my_recipes/rllm/tools/tool_base.py` | 1 | abc, camel.toolkits, dataclasses, inspect, os |
| `my_recipes/rllm/tools/utils` | `my_recipes/rllm/tools/utils.py` | 0 | asyncio, inspect, json, openai, typing |
| `my_recipes/rllm/tools/web_tools/__init__` | `my_recipes/rllm/tools/web_tools/__init__.py` | 3 | os |
| `my_recipes/rllm/tools/web_tools/firecrawl_tool` | `my_recipes/rllm/tools/web_tools/firecrawl_tool.py` | 1 | asyncio, firecrawl, os, time, typing |
| `my_recipes/rllm/tools/web_tools/gsearch_tool` | `my_recipes/rllm/tools/web_tools/gsearch_tool.py` | 1 | httpx, os, typing |
| `my_recipes/rllm/tools/web_tools/tavily_tool` | `my_recipes/rllm/tools/web_tools/tavily_tool.py` | 1 | asyncio, httpx, os, typing |
| `my_recipes/rllm/utils` | `my_recipes/rllm/utils.py` | 1 | google.cloud.aiplatform_v1beta1.types.content, openai, os, random, sentence_transformers |
| `my_recipes/syntool/__init__` | `my_recipes/syntool/__init__.py` | 4 | - |
| `my_recipes/syntool/agent_loop` | `my_recipes/syntool/agent_loop.py` | 4 | __future__, json, numpy, typing |
| `my_recipes/syntool/dataset` | `my_recipes/syntool/dataset.py` | 2 | __future__, copy, json, logging, numpy |
| `my_recipes/syntool/reward` | `my_recipes/syntool/reward.py` | 0 | __future__, re, typing |
| `my_recipes/syntool/tool` | `my_recipes/syntool/tool.py` | 3 | __future__, traceback, typing |
| `scripts/__init__` | `scripts/__init__.py` | 0 | - |
| `scripts/converter_hf_to_mcore` | `scripts/converter_hf_to_mcore.py` | 6 | accelerate, argparse, contextlib, importlib.metadata, megatron.core.dist_checkpointing.mapping |
| `scripts/diagnose` | `scripts/diagnose.py` | 0 | argparse, importlib.metadata, os, pip, platform |
| `scripts/init_random_model` | `scripts/init_random_model.py` | 0 | argparse, json, os, transformers, typing |
| `scripts/legacy_model_merger` | `scripts/legacy_model_merger.py` | 3 | abc, accelerate, argparse, concurrent.futures, dataclasses |
| `scripts/megatron_merge_lora` | `scripts/megatron_merge_lora.py` | 3 | hydra, omegaconf, os, pprint, ray |
| `scripts/print_cfg` | `scripts/print_cfg.py` | 1 | hydra |
| `scripts/rollout_viewer` | `scripts/rollout_viewer.py` | 0 | aiofiles, asyncio, json, packaging.version, pathlib |
| `scripts/veomni/moe_merge` | `scripts/veomni/moe_merge.py` | 0 | argparse, dataclasses, glob, os, safetensors.torch |
| `scripts/veomni/moe_split` | `scripts/veomni/moe_split.py` | 0 | argparse, dataclasses, glob, os, safetensors.torch |
| `setup` | `setup.py` | 0 | os, pathlib, setuptools |
| `tests/__init__` | `tests/__init__.py` | 0 | - |
| `tests/checkpoint_engine/__init__` | `tests/checkpoint_engine/__init__.py` | 0 | - |
| `tests/checkpoint_engine/test_correctness_on_gpu` | `tests/checkpoint_engine/test_correctness_on_gpu.py` | 5 | os, pytest, ray, verl.checkpoint_engine |
| `tests/checkpoint_engine/test_correctness_on_npu` | `tests/checkpoint_engine/test_correctness_on_npu.py` | 5 | os, pytest, ray, verl.checkpoint_engine |
| `tests/checkpoint_engine/test_special_server_adapter` | `tests/checkpoint_engine/test_special_server_adapter.py` | 5 | asyncio, hydra, omegaconf, os, pytest |
| `tests/checkpoint_engine/test_utils` | `tests/checkpoint_engine/test_utils.py` | 6 | asyncio, ray, torch, transformers, typing |
| `tests/experimental/agent_loop/agent_utils` | `tests/experimental/agent_loop/agent_utils.py` | 6 | omegaconf, ray, verl.checkpoint_engine, verl.single_controller.ray |
| `tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu` | `tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py` | 4 | __future__, numpy, omegaconf, pytest, torch |
| `tests/experimental/agent_loop/test_basic_agent_loop` | `tests/experimental/agent_loop/test_basic_agent_loop.py` | 8 | hydra, json, numpy, omegaconf, os |
| `tests/experimental/agent_loop/test_gpt_oss_tool_parser` | `tests/experimental/agent_loop/test_gpt_oss_tool_parser.py` | 1 | pytest, transformers |
| `tests/experimental/agent_loop/test_multi_modal` | `tests/experimental/agent_loop/test_multi_modal.py` | 5 | PIL, hydra, json, numpy, omegaconf |
| `tests/experimental/agent_loop/test_standalone_rollout` | `tests/experimental/agent_loop/test_standalone_rollout.py` | 3 | asyncio, hydra, omegaconf, openai, os |
| `tests/experimental/reward_loop/reward_fn` | `tests/experimental/reward_loop/reward_fn.py` | 1 | aiohttp, json, openai.types.chat, os, transformers |
| `tests/experimental/reward_loop/test_agent_reward_loop_colocate` | `tests/experimental/reward_loop/test_agent_reward_loop_colocate.py` | 9 | hydra, os, ray, torchdata.stateful_dataloader, transformers |
| `tests/experimental/reward_loop/test_agent_reward_loop_standalone` | `tests/experimental/reward_loop/test_agent_reward_loop_standalone.py` | 6 | hydra, os, ray, torchdata.stateful_dataloader |
| `tests/experimental/reward_loop/test_async_token_bucket_on_cpu` | `tests/experimental/reward_loop/test_async_token_bucket_on_cpu.py` | 1 | asyncio, pytest, time |
| `tests/experimental/reward_loop/test_math_verify` | `tests/experimental/reward_loop/test_math_verify.py` | 4 | hydra, os, ray, torchdata.stateful_dataloader, transformers |
| `tests/experimental/reward_loop/test_rate_limited_reward_manager_on_cpu` | `tests/experimental/reward_loop/test_rate_limited_reward_manager_on_cpu.py` | 1 | asyncio, omegaconf, os.path, pytest, time |
| `tests/experimental/reward_loop/test_reward_model_disrm` | `tests/experimental/reward_loop/test_reward_model_disrm.py` | 5 | hydra, os, ray, torch |
| `tests/experimental/reward_loop/test_reward_model_genrm` | `tests/experimental/reward_loop/test_reward_model_genrm.py` | 5 | hydra, os, ray, torch |
| `tests/experimental/reward_loop/test_visual_reward_manager` | `tests/experimental/reward_loop/test_visual_reward_manager.py` | 3 | hydra, os, ray, torch |
| `tests/experimental/vla/test_sim_envs` | `tests/experimental/vla/test_sim_envs.py` | 2 | numpy, omegaconf, os, pytest, unittest |
| `tests/interactions/__init__` | `tests/interactions/__init__.py` | 0 | - |
| `tests/interactions/test_gsm8k_interaction` | `tests/interactions/test_gsm8k_interaction.py` | 2 | pytest, unittest.mock |
| `tests/interactions/test_interaction_registry` | `tests/interactions/test_interaction_registry.py` | 3 | omegaconf, os, pytest, tempfile |
| `tests/models/test_engine` | `tests/models/test_engine.py` | 9 | functools, numpy, os, pytest, ray |
| `tests/models/test_liger_vl_compat` | `tests/models/test_liger_vl_compat.py` | 1 | pytest, torch, transformers |
| `tests/models/test_tiled_mlp_accuracy` | `tests/models/test_tiled_mlp_accuracy.py` | 2 | torch, torch.distributed.device_mesh, torch.distributed.fsdp, transformers |
| `tests/models/test_transformer` | `tests/models/test_transformer.py` | 4 | flash_attn.bert_padding, torch, transformers |
| `tests/models/test_transformers_ulysses` | `tests/models/test_transformers_ulysses.py` | 8 | contextlib, copy, dataclasses, flash_attn.bert_padding, packaging |
| `tests/my_recipes/test_syntool_recipe` | `tests/my_recipes/test_syntool_recipe.py` | 8 | __future__, json, omegaconf, pandas, pathlib |
| `tests/single_controller/__init__` | `tests/single_controller/__init__.py` | 0 | - |
| `tests/single_controller/base/test_decorator` | `tests/single_controller/base/test_decorator.py` | 1 | pytest |
| `tests/single_controller/check_worker_alive/main` | `tests/single_controller/check_worker_alive/main.py` | 3 | os, ray, sys, time |
| `tests/single_controller/detached_worker/client` | `tests/single_controller/detached_worker/client.py` | 1 | ray, tensordict, torch, verl, verl.single_controller.ray |
| `tests/single_controller/detached_worker/server` | `tests/single_controller/detached_worker/server.py` | 5 | megatron.core.models.gpt.gpt_model, omegaconf, os, ray, tensordict |
| `tests/single_controller/test_auto_padding_on_cpu` | `tests/single_controller/test_auto_padding_on_cpu.py` | 4 | numpy, ray, torch, verl |
| `tests/single_controller/test_colocated_workers` | `tests/single_controller/test_colocated_workers.py` | 4 | ray, torch, verl |
| `tests/single_controller/test_colocated_workers_fused` | `tests/single_controller/test_colocated_workers_fused.py` | 4 | ray, torch, verl |
| `tests/single_controller/test_data_transfer` | `tests/single_controller/test_data_transfer.py` | 4 | codetiming, packaging, pickle, ray, tensordict |
| `tests/single_controller/test_decorator_on_cpu` | `tests/single_controller/test_decorator_on_cpu.py` | 4 | asyncio, pytest, ray, tensordict, time |
| `tests/single_controller/test_device_mesh_register` | `tests/single_controller/test_device_mesh_register.py` | 5 | numpy, ray, tensordict, torch, verl |
| `tests/single_controller/test_driverfunc_to_worker` | `tests/single_controller/test_driverfunc_to_worker.py` | 3 | os, ray, tensordict, torch, verl |
| `tests/single_controller/test_fused_workers_on_cpu` | `tests/single_controller/test_fused_workers_on_cpu.py` | 3 | ray |
| `tests/single_controller/test_get_set_dispatch_collect_cpu` | `tests/single_controller/test_get_set_dispatch_collect_cpu.py` | 1 | os, pytest |
| `tests/single_controller/test_high_level_scheduling_api` | `tests/single_controller/test_high_level_scheduling_api.py` | 3 | gc, ray, time |
| `tests/single_controller/test_nested_worker` | `tests/single_controller/test_nested_worker.py` | 4 | ray |
| `tests/single_controller/test_ray_collectives` | `tests/single_controller/test_ray_collectives.py` | 2 | ray, ray.util.collective, torch, verl.single_controller.ray |
| `tests/single_controller/test_ray_local_envs_on_cpu` | `tests/single_controller/test_ray_local_envs_on_cpu.py` | 2 | os, ray |
| `tests/single_controller/test_ray_utils_on_cpu` | `tests/single_controller/test_ray_utils_on_cpu.py` | 1 | pytest, ray |
| `tests/single_controller/test_rvdz` | `tests/single_controller/test_rvdz.py` | 1 | ray |
| `tests/single_controller/test_split_resource_pool` | `tests/single_controller/test_split_resource_pool.py` | 4 | os, ray, torch, verl |
| `tests/single_controller/test_worker_group_basics` | `tests/single_controller/test_worker_group_basics.py` | 4 | ray, torch |
| `tests/single_controller/test_worker_group_torch` | `tests/single_controller/test_worker_group_torch.py` | 4 | os, ray, torch |
| `tests/special_distributed/test_fsdp_ckpt` | `tests/special_distributed/test_fsdp_ckpt.py` | 6 | flash_attn.bert_padding, os, shutil, tempfile, torch |
| `tests/special_distributed/test_mcore_config_converter` | `tests/special_distributed/test_mcore_config_converter.py` | 1 | megatron.core.parallel_state, megatron.core.transformer, os, torch, transformers |
| `tests/special_distributed/test_tensor_dict` | `tests/special_distributed/test_tensor_dict.py` | 6 | numpy, os, torch, verl.utils.profiler |
| `tests/special_distributed/test_torch_functional` | `tests/special_distributed/test_torch_functional.py` | 1 | os, torch |
| `tests/special_e2e/__init__` | `tests/special_e2e/__init__.py` | 0 | - |
| `tests/special_e2e/check_custom_rwd_fn` | `tests/special_e2e/check_custom_rwd_fn.py` | 0 | argparse |
| `tests/special_e2e/check_results` | `tests/special_e2e/check_results.py` | 0 | argparse, numpy |
| `tests/special_e2e/envs/__init__` | `tests/special_e2e/envs/__init__.py` | 0 | .digit_completion |
| `tests/special_e2e/envs/digit_completion/__init__` | `tests/special_e2e/envs/digit_completion/__init__.py` | 2 | transformers |
| `tests/special_e2e/envs/digit_completion/task` | `tests/special_e2e/envs/digit_completion/task.py` | 0 | numpy |
| `tests/special_e2e/envs/digit_completion/tokenizer` | `tests/special_e2e/envs/digit_completion/tokenizer.py` | 0 | json, os, pathlib, transformers.tokenization_utils, typing |
| `tests/special_e2e/sft/compare_sft_engine_results` | `tests/special_e2e/sft/compare_sft_engine_results.py` | 0 | json, os, torch |
| `tests/special_sanity/check_api_docs` | `tests/special_sanity/check_api_docs.py` | 0 | __future__, argparse, importlib, inspect, pathlib |
| `tests/special_sanity/check_dataproto_usage` | `tests/special_sanity/check_dataproto_usage.py` | 0 | argparse, os, pathlib |
| `tests/special_sanity/check_device_api_usage` | `tests/special_sanity/check_device_api_usage.py` | 0 | argparse, os, pathlib |
| `tests/special_sanity/check_docs_time_info` | `tests/special_sanity/check_docs_time_info.py` | 0 | pathlib, sys |
| `tests/special_sanity/check_docstrings` | `tests/special_sanity/check_docstrings.py` | 0 | ast, os, sys |
| `tests/special_sanity/check_license` | `tests/special_sanity/check_license.py` | 0 | argparse, pathlib, typing |
| `tests/special_sanity/check_pr_description` | `tests/special_sanity/check_pr_description.py` | 0 | json, os |
| `tests/special_sanity/check_pr_title` | `tests/special_sanity/check_pr_title.py` | 0 | os, re |
| `tests/special_sanity/test_config_docs` | `tests/special_sanity/test_config_docs.py` | 0 | pathlib, re |
| `tests/special_sanity/test_import` | `tests/special_sanity/test_import.py` | 0 | verl, verl.single_controller |
| `tests/special_sanity/type_coverage_check` | `tests/special_sanity/type_coverage_check.py` | 0 | argparse, ast, linecache, pathlib, subprocess |
| `tests/special_sanity/validate_imported_docs` | `tests/special_sanity/validate_imported_docs.py` | 0 | __future__, argparse, ast, importlib, inspect |
| `tests/special_sanity/validate_structure` | `tests/special_sanity/validate_structure.py` | 0 | __future__, argparse, pathlib, sys |
| `tests/special_standalone/test_memory_buffers` | `tests/special_standalone/test_memory_buffers.py` | 0 | gc, torch, transformers |
| `tests/test_base_config_on_cpu` | `tests/test_base_config_on_cpu.py` | 1 | pytest |
| `tests/test_protocol_on_cpu` | `tests/test_protocol_on_cpu.py` | 2 | numpy, os, packaging.version, pickle, pytest |
| `tests/test_protocol_v2_on_cpu` | `tests/test_protocol_v2_on_cpu.py` | 1 | copy, numpy, os, pytest, random |
| `tests/trainer/__init__` | `tests/trainer/__init__.py` | 0 | - |
| `tests/trainer/config/__init__` | `tests/trainer/config/__init__.py` | 0 | - |
| `tests/trainer/config/test_algo_config_on_cpu` | `tests/trainer/config/test_algo_config_on_cpu.py` | 3 | hydra, numpy, omegaconf, os, torch |
| `tests/trainer/config/test_legacy_config_on_cpu` | `tests/trainer/config/test_legacy_config_on_cpu.py` | 0 | hydra, hydra.core.global_hydra, omegaconf, os, unittest |
| `tests/trainer/ppo/__init__` | `tests/trainer/ppo/__init__.py` | 0 | - |
| `tests/trainer/ppo/test_core_algos_on_cpu` | `tests/trainer/ppo/test_core_algos_on_cpu.py` | 1 | enum, numpy, pytest, random, torch |
| `tests/trainer/ppo/test_metric_utils_on_cpu` | `tests/trainer/ppo/test_metric_utils_on_cpu.py` | 2 | numpy, torch, unittest, unittest.mock, verl.utils.metric |
| `tests/trainer/ppo/test_rollout_corr` | `tests/trainer/ppo/test_rollout_corr.py` | 1 | pytest, torch, traceback |
| `tests/trainer/ppo/test_rollout_corr_integration` | `tests/trainer/ppo/test_rollout_corr_integration.py` | 5 | pytest, torch |
| `tests/utils/_test_module` | `tests/utils/_test_module.py` | 0 | - |
| `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu` | `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py` | 2 | os, pytest, shutil, tempfile |
| `tests/utils/ckpt/test_esi_save_ckpt_on_cpu` | `tests/utils/ckpt/test_esi_save_ckpt_on_cpu.py` | 1 | datetime, os, time, unittest |
| `tests/utils/dataset/test_create_rl_sampler_on_cpu` | `tests/utils/dataset/test_create_rl_sampler_on_cpu.py` | 2 | collections.abc, omegaconf, pytest, torch, torch.utils.data |
| `tests/utils/dataset/test_multiturn_sft_dataset_on_cpu` | `tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py` | 4 | PIL, io, os, pandas, pathlib |
| `tests/utils/dataset/test_rl_collate_fn_on_cpu` | `tests/utils/dataset/test_rl_collate_fn_on_cpu.py` | 1 | torch |
| `tests/utils/dataset/test_rl_dataset_on_cpu` | `tests/utils/dataset/test_rl_dataset_on_cpu.py` | 2 | PIL, json, omegaconf, os, pytest |
| `tests/utils/debug/test_metrics` | `tests/utils/debug/test_metrics.py` | 2 | torch, unittest |
| `tests/utils/megatron/test_pipeline_parallel` | `tests/utils/megatron/test_pipeline_parallel.py` | 2 | pytest |
| `tests/utils/reward_score/reward_score/test_sandbox_fusion_on_cpu` | `tests/utils/reward_score/reward_score/test_sandbox_fusion_on_cpu.py` | 1 | concurrent.futures, multiprocessing, os, pytest, sys |
| `tests/utils/reward_score/test_sandbox_on_cpu` | `tests/utils/reward_score/test_sandbox_on_cpu.py` | 1 | asyncio, collections, json, os, pytest |
| `tests/utils/test_activation_offload` | `tests/utils/test_activation_offload.py` | 7 | flash_attn.bert_padding, os, pytest, shutil, tempfile |
| `tests/utils/test_bucketed_weight_transfer` | `tests/utils/test_bucketed_weight_transfer.py` | 2 | asyncio, multiprocessing, pytest, torch, uuid |
| `tests/utils/test_check_ipc_version_support_on_npu` | `tests/utils/test_check_ipc_version_support_on_npu.py` | 1 | logging, unittest, unittest.mock |
| `tests/utils/test_check_profiler_output` | `tests/utils/test_check_profiler_output.py` | 0 | argparse, dataclasses, glob, logging, os |
| `tests/utils/test_config_on_cpu` | `tests/utils/test_config_on_cpu.py` | 2 | dataclasses, omegaconf, subprocess, unittest |
| `tests/utils/test_flops_counter` | `tests/utils/test_flops_counter.py` | 1 | math, pytest |
| `tests/utils/test_fs_on_cpu` | `tests/utils/test_fs_on_cpu.py` | 1 | os, pathlib |
| `tests/utils/test_fsdp2_peft_wrapping` | `tests/utils/test_fsdp2_peft_wrapping.py` | 1 | torch.nn, types, unittest |
| `tests/utils/test_fsdp_lora_merge` | `tests/utils/test_fsdp_lora_merge.py` | 3 | os, peft, peft.tuners.lora, peft.utils.save_and_load, pytest |
| `tests/utils/test_groupwise` | `tests/utils/test_groupwise.py` | 2 | numpy, os, pytest, torch |
| `tests/utils/test_import_utils_on_cpu` | `tests/utils/test_import_utils_on_cpu.py` | 1 | os, pytest, tempfile |
| `tests/utils/test_linear_cross_entropy` | `tests/utils/test_linear_cross_entropy.py` | 4 | gc, os, torch |
| `tests/utils/test_mlflow_key_sanitization` | `tests/utils/test_mlflow_key_sanitization.py` | 1 | unittest, unittest.mock |
| `tests/utils/test_model_on_cpu` | `tests/utils/test_model_on_cpu.py` | 1 | pytest, types |
| `tests/utils/test_normalize_peft_param_name` | `tests/utils/test_normalize_peft_param_name.py` | 4 | os, peft, pytest, torch, torch.distributed.fsdp |
| `tests/utils/test_normalize_peft_param_name_on_cpu` | `tests/utils/test_normalize_peft_param_name_on_cpu.py` | 1 | peft, pytest, torch, transformers |
| `tests/utils/test_nvtx_profile` | `tests/utils/test_nvtx_profile.py` | 3 | dataclasses, hydra, os, unittest, unittest.mock |
| `tests/utils/test_padding_on_cpu` | `tests/utils/test_padding_on_cpu.py` | 1 | tensordict, torch |
| `tests/utils/test_prepare_micro_batches_with_group_size` | `tests/utils/test_prepare_micro_batches_with_group_size.py` | 2 | tensordict, torch |
| `tests/utils/test_rollout_skip_on_cpu` | `tests/utils/test_rollout_skip_on_cpu.py` | 2 | numpy, omegaconf, pathlib, pytest, shutil |
| `tests/utils/test_rollout_trace_on_cpu` | `tests/utils/test_rollout_trace_on_cpu.py` | 1 | os, pytest, sys, unittest.mock |
| `tests/utils/test_seqlen_balancing` | `tests/utils/test_seqlen_balancing.py` | 4 | numpy, torch, torch.multiprocessing, verl |
| `tests/utils/test_server_profiler` | `tests/utils/test_server_profiler.py` | 3 | json, os, unittest, unittest.mock |
| `tests/utils/test_shared_memory` | `tests/utils/test_shared_memory.py` | 1 | multiprocessing, torch, unittest, uuid |
| `tests/utils/test_special_linear_cross_entropy_tp` | `tests/utils/test_special_linear_cross_entropy_tp.py` | 3 | gc, os, sys, torch |
| `tests/utils/test_special_megatron_kl_loss_tp` | `tests/utils/test_special_megatron_kl_loss_tp.py` | 4 | gc, megatron.core.parallel_state, os, torch, torch.nn.functional |
| `tests/utils/test_special_mstx_profile` | `tests/utils/test_special_mstx_profile.py` | 3 | unittest, unittest.mock |
| `tests/utils/test_temp_env_on_cpu` | `tests/utils/test_temp_env_on_cpu.py` | 1 | os, pytest |
| `tests/utils/test_timeout_decorator_cpu` | `tests/utils/test_timeout_decorator_cpu.py` | 1 | multiprocessing, pytest, sys, threading, time |
| `tests/utils/test_tokenizer_normalize_on_cpu` | `tests/utils/test_tokenizer_normalize_on_cpu.py` | 1 | numpy, pytest |
| `tests/utils/test_torch_functional` | `tests/utils/test_torch_functional.py` | 3 | os, pytest, torch, torch.multiprocessing |
| `tests/utils/test_torch_profile` | `tests/utils/test_torch_profile.py` | 2 | torch, unittest, unittest.mock |
| `tests/workers/actor/test_special_dp_actor` | `tests/workers/actor/test_special_dp_actor.py` | 3 | tensordict, torch, torch.nn, transformers, unittest |
| `tests/workers/config/test_actor_config_on_cpu` | `tests/workers/config/test_actor_config_on_cpu.py` | 2 | hydra, os, unittest |
| `tests/workers/config/test_critic_config_on_cpu` | `tests/workers/config/test_critic_config_on_cpu.py` | 2 | hydra, os, pathlib, pytest, verl.utils.profiler |
| `tests/workers/config/test_engine_config_on_cpu` | `tests/workers/config/test_engine_config_on_cpu.py` | 1 | pytest |
| `tests/workers/config/test_model_config_on_cpu` | `tests/workers/config/test_model_config_on_cpu.py` | 1 | omegaconf, os, pytest |
| `tests/workers/config/test_optim_config_on_cpu` | `tests/workers/config/test_optim_config_on_cpu.py` | 1 | pytest |
| `tests/workers/critic/test_special_dp_critic` | `tests/workers/critic/test_special_dp_critic.py` | 5 | omegaconf, os, shutil, tempfile, tensordict |
| `tests/workers/reward_manager/test_registry_on_cpu` | `tests/workers/reward_manager/test_registry_on_cpu.py` | 1 | pytest |
| `tests/workers/rollout/perf/vllm_async_rollout` | `tests/workers/rollout/perf/vllm_async_rollout.py` | 5 | hydra, omegaconf, os, ray, time |
| `tests/workers/rollout/rollout_sglang/test_http_server_engine` | `tests/workers/rollout/rollout_sglang/test_http_server_engine.py` | 2 | aiohttp, asyncio, base64, itertools, os |
| `tests/workers/rollout/rollout_sglang/test_lora_sleep_level` | `tests/workers/rollout/rollout_sglang/test_lora_sleep_level.py` | 0 | __future__, asyncio, dataclasses, typing, unittest.mock |
| `tests/workers/rollout/rollout_trtllm/__init__` | `tests/workers/rollout/rollout_trtllm/__init__.py` | 0 | - |
| `tests/workers/rollout/rollout_trtllm/test_adapter` | `tests/workers/rollout/rollout_trtllm/test_adapter.py` | 2 | aiohttp, asyncio, hydra, os, pytest |
| `tests/workers/rollout/rollout_trtllm/test_async_server` | `tests/workers/rollout/rollout_trtllm/test_async_server.py` | 3 | hydra, os, ray, ray.util.scheduling_strategies, subprocess |
| `tests/workers/rollout/rollout_trtllm/test_trtllm_rollout_utils` | `tests/workers/rollout/rollout_trtllm/test_trtllm_rollout_utils.py` | 1 | PIL, asyncio, numpy, omegaconf, pytest |
| `tests/workers/rollout/rollout_vllm/run_fsdp_vllm` | `tests/workers/rollout/rollout_vllm/run_fsdp_vllm.py` | 3 | os, time, torch, torch.distributed.device_mesh, torch.distributed.fsdp |
| `tests/workers/rollout/rollout_vllm/test_vllm_abort` | `tests/workers/rollout/rollout_vllm/test_vllm_abort.py` | 2 | asyncio, hydra, os, ray, time |
| `tests/workers/rollout/rollout_vllm/test_vllm_omni_generate` | `tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py` | 3 | omegaconf, os, pathlib, pytest, ray |
| `tests/workers/rollout/test_hf_rollout` | `tests/workers/rollout/test_hf_rollout.py` | 4 | omegaconf, os, torch, torch.distributed.device_mesh, torch.distributed.fsdp |
| `tests/workers/rollout/test_sglang_async_rollout_multimodal_delta` | `tests/workers/rollout/test_sglang_async_rollout_multimodal_delta.py` | 4 | os, pytest |
| `tests/workers/rollout/test_sglang_rollout_sharding_manager` | `tests/workers/rollout/test_sglang_rollout_sharding_manager.py` | 1 | pytest, torch |
| `tests/workers/rollout/test_vllm_cli_args_on_cpu` | `tests/workers/rollout/test_vllm_cli_args_on_cpu.py` | 1 | json, pytest |
| `tests/workers/test_fsdp_attn_implementation` | `tests/workers/test_fsdp_attn_implementation.py` | 2 | omegaconf, pytest, torch, transformers, unittest.mock |
| `tests/workers/test_fsdp_workers` | `tests/workers/test_fsdp_workers.py` | 1 | omegaconf, os |
| `verl/__init__` | `verl/__init__.py` | 5 | importlib, logging, modelscope.utils.hf_util, os, packaging.version |
| `verl/base_config` | `verl/base_config.py` | 0 | collections, dataclasses, typing |
| `verl/checkpoint_engine/__init__` | `verl/checkpoint_engine/__init__.py` | 6 | - |
| `verl/checkpoint_engine/base` | `verl/checkpoint_engine/base.py` | 6 | abc, asyncio, ray, torch, typing |
| `verl/checkpoint_engine/hccl_checkpoint_engine` | `verl/checkpoint_engine/hccl_checkpoint_engine.py` | 5 | dataclasses, logging, os, ray, time |
| `verl/checkpoint_engine/kimi_checkpoint_engine` | `verl/checkpoint_engine/kimi_checkpoint_engine.py` | 4 | asyncio, checkpoint_engine.ps, collections, concurrent.futures, dataclasses |
| `verl/checkpoint_engine/mooncake_checkpoint_engine` | `verl/checkpoint_engine/mooncake_checkpoint_engine.py` | 5 | asyncio, gc, logging, os, ray |
| `verl/checkpoint_engine/nccl_checkpoint_engine` | `verl/checkpoint_engine/nccl_checkpoint_engine.py` | 2 | asyncio, cupy, dataclasses, logging, os |
| `verl/checkpoint_engine/nixl_checkpoint_engine` | `verl/checkpoint_engine/nixl_checkpoint_engine.py` | 2 | asyncio, collections, cupy, dataclasses, logging |
| `verl/experimental/__init__` | `verl/experimental/__init__.py` | 0 | - |
| `verl/experimental/agent_loop/__init__` | `verl/experimental/agent_loop/__init__.py` | 3 | - |
| `verl/experimental/agent_loop/agent_loop` | `verl/experimental/agent_loop/agent_loop.py` | 15 | PIL, abc, asyncio, cachetools, hydra |
| `verl/experimental/agent_loop/prometheus_utils` | `verl/experimental/agent_loop/prometheus_utils.py` | 1 | logging, os, ray, socket, subprocess |
| `verl/experimental/agent_loop/single_turn_agent_loop` | `verl/experimental/agent_loop/single_turn_agent_loop.py` | 2 | logging, os, typing, uuid, verl.utils.profiler |
| `verl/experimental/agent_loop/tool_agent_loop` | `verl/experimental/agent_loop/tool_agent_loop.py` | 9 | PIL, asyncio, enum, json, logging |
| `verl/experimental/agent_loop/tool_parser` | `verl/experimental/agent_loop/tool_parser.py` | 3 | abc, json, logging, os, pydantic |
| `verl/experimental/agent_loop/utils` | `verl/experimental/agent_loop/utils.py` | 0 | os, typing, verl |
| `verl/experimental/dataset/__init__` | `verl/experimental/dataset/__init__.py` | 0 | - |
| `verl/experimental/dataset/sampler` | `verl/experimental/dataset/sampler.py` | 0 | abc, collections.abc, omegaconf, torch.utils.data, verl |
| `verl/experimental/dynamic_dataset/__init__` | `verl/experimental/dynamic_dataset/__init__.py` | 0 | - |
| `verl/experimental/dynamic_dataset/dynamicgen_dataset` | `verl/experimental/dynamic_dataset/dynamicgen_dataset.py` | 2 | abc, datasets, logging, omegaconf, torch.utils.data |
| `verl/experimental/fully_async_policy/agent_loop/__init__` | `verl/experimental/fully_async_policy/agent_loop/__init__.py` | 1 | - |
| `verl/experimental/fully_async_policy/agent_loop/agent_loop` | `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` | 4 | asyncio, logging, omegaconf, os, ray |
| `verl/experimental/fully_async_policy/detach_utils` | `verl/experimental/fully_async_policy/detach_utils.py` | 1 | asyncio, collections, dataclasses, numpy, time |
| `verl/experimental/fully_async_policy/fully_async_main` | `verl/experimental/fully_async_policy/fully_async_main.py` | 9 | asyncio, concurrent.futures, hydra, omegaconf, os |
| `verl/experimental/fully_async_policy/fully_async_rollouter` | `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 11 | asyncio, concurrent.futures, multiprocessing, numpy, os |
| `verl/experimental/fully_async_policy/fully_async_trainer` | `verl/experimental/fully_async_policy/fully_async_trainer.py` | 11 | datetime, logging, omegaconf, os, pprint |
| `verl/experimental/fully_async_policy/message_queue` | `verl/experimental/fully_async_policy/message_queue.py` | 0 | asyncio, collections, logging, omegaconf, ray |
| `verl/experimental/fully_async_policy/unittest/simple_streaming_demo` | `verl/experimental/fully_async_policy/unittest/simple_streaming_demo.py` | 0 | asyncio, random, time |
| `verl/experimental/one_step_off_policy/main_ppo` | `verl/experimental/one_step_off_policy/main_ppo.py` | 9 | asyncio, hydra, omegaconf, os, pprint |
| `verl/experimental/one_step_off_policy/ray_trainer` | `verl/experimental/one_step_off_policy/ray_trainer.py` | 7 | asyncio, numpy, omegaconf, pprint, ray |
| `verl/experimental/reward_loop/__init__` | `verl/experimental/reward_loop/__init__.py` | 2 | - |
| `verl/experimental/reward_loop/reward_loop` | `verl/experimental/reward_loop/reward_loop.py` | 8 | PIL, aiohttp, asyncio, logging, numpy |
| `verl/experimental/reward_loop/reward_manager/__init__` | `verl/experimental/reward_loop/reward_manager/__init__.py` | 7 | - |
| `verl/experimental/reward_loop/reward_manager/base` | `verl/experimental/reward_loop/reward_manager/base.py` | 1 | abc, logging, omegaconf, os, transformers |
| `verl/experimental/reward_loop/reward_manager/dapo` | `verl/experimental/reward_loop/reward_manager/dapo.py` | 1 | inspect, verl, verl.experimental.reward_loop.reward_manager, verl.utils.reward_score |
| `verl/experimental/reward_loop/reward_manager/gdpo` | `verl/experimental/reward_loop/reward_manager/gdpo.py` | 1 | inspect, verl, verl.experimental.reward_loop.reward_manager, verl.utils.reward_score |
| `verl/experimental/reward_loop/reward_manager/limited` | `verl/experimental/reward_loop/reward_manager/limited.py` | 2 | asyncio, collections, inspect, logging, omegaconf |
| `verl/experimental/reward_loop/reward_manager/naive` | `verl/experimental/reward_loop/reward_manager/naive.py` | 1 | inspect, verl, verl.experimental.reward_loop.reward_manager, verl.utils.reward_score |
| `verl/experimental/reward_loop/reward_manager/registry` | `verl/experimental/reward_loop/reward_manager/registry.py` | 1 | typing |
| `verl/experimental/reward_loop/reward_manager/remote` | `verl/experimental/reward_loop/reward_manager/remote.py` | 1 | inspect, itertools, ray, verl, verl.experimental.reward_loop.reward_manager |
| `verl/experimental/reward_loop/reward_manager/visual` | `verl/experimental/reward_loop/reward_manager/visual.py` | 1 | inspect, verl, verl.experimental.reward_loop.reward_manager, verl.utils.reward_score |
| `verl/experimental/reward_loop/reward_model` | `verl/experimental/reward_loop/reward_model.py` | 4 | asyncio, logging, os |
| `verl/experimental/reward_loop/router/inner_sglang_router` | `verl/experimental/reward_loop/router/inner_sglang_router.py` | 1 | logging, multiprocessing, os, ray, requests |
| `verl/experimental/reward_loop/router/naive_router` | `verl/experimental/reward_loop/router/naive_router.py` | 1 | aiohttp, asyncio, fastapi, fastapi.responses, logging |
| `verl/experimental/separation/__init__` | `verl/experimental/separation/__init__.py` | 0 | - |
| `verl/experimental/separation/engine_workers` | `verl/experimental/separation/engine_workers.py` | 5 | logging, omegaconf, os |
| `verl/experimental/separation/ray_trainer` | `verl/experimental/separation/ray_trainer.py` | 17 | copy, functools, numpy, omegaconf, pprint |
| `verl/experimental/separation/utils` | `verl/experimental/separation/utils.py` | 4 | ray, verl.single_controller.ray |
| `verl/experimental/teacher_loop/__init__` | `verl/experimental/teacher_loop/__init__.py` | 1 | - |
| `verl/experimental/teacher_loop/teacher_manager` | `verl/experimental/teacher_loop/teacher_manager.py` | 5 | asyncio, omegaconf, ray, tensordict, torch |
| `verl/experimental/teacher_loop/teacher_model` | `verl/experimental/teacher_loop/teacher_model.py` | 8 | asyncio, logging, omegaconf, os |
| `verl/experimental/vla/dp_rob` | `verl/experimental/vla/dp_rob.py` | 7 | logging, torch, torch.distributed.fsdp, verl.trainer.ppo |
| `verl/experimental/vla/env_loop` | `verl/experimental/vla/env_loop.py` | 0 | asyncio, logging, numpy, omegaconf, os |
| `verl/experimental/vla/envs/__init__` | `verl/experimental/vla/envs/__init__.py` | 0 | - |
| `verl/experimental/vla/envs/action_utils` | `verl/experimental/vla/envs/action_utils.py` | 1 | PIL, imageio, io, numpy, os |
| `verl/experimental/vla/envs/isaac_env/__init__` | `verl/experimental/vla/envs/isaac_env/__init__.py` | 1 | - |
| `verl/experimental/vla/envs/isaac_env/isaac_env` | `verl/experimental/vla/envs/isaac_env/isaac_env.py` | 2 | gymnasium, isaaclab.app, isaaclab_playground.tasks.manipulation.libero.config.franka, logging, numpy |
| `verl/experimental/vla/envs/libero_env/__init__` | `verl/experimental/vla/envs/libero_env/__init__.py` | 0 | - |
| `verl/experimental/vla/envs/libero_env/libero_env` | `verl/experimental/vla/envs/libero_env/libero_env.py` | 3 | gymnasium, libero.libero, libero.libero.benchmark, libero.libero.envs, logging |
| `verl/experimental/vla/envs/libero_env/utils` | `verl/experimental/vla/envs/libero_env/utils.py` | 0 | math, numpy |
| `verl/experimental/vla/envs/libero_env/venv` | `verl/experimental/vla/envs/libero_env/venv.py` | 1 | gymnasium, libero.libero.envs, multiprocessing, multiprocessing.context, numpy |
| `verl/experimental/vla/fsdp_workers` | `verl/experimental/vla/fsdp_workers.py` | 16 | asyncio, contextlib, logging, omegaconf, os |
| `verl/experimental/vla/main_ppo` | `verl/experimental/vla/main_ppo.py` | 10 | datasets, hydra, logging, omegaconf, pprint |
| `verl/experimental/vla/main_sac` | `verl/experimental/vla/main_sac.py` | 8 | datasets, hydra, logging, omegaconf, pprint |
| `verl/experimental/vla/models/__init__` | `verl/experimental/vla/models/__init__.py` | 1 | - |
| `verl/experimental/vla/models/modules/mlp` | `verl/experimental/vla/models/modules/mlp.py` | 0 | torch.nn, torch.nn.init |
| `verl/experimental/vla/models/openvla_oft/__init__` | `verl/experimental/vla/models/openvla_oft/__init__.py` | 0 | - |
| `verl/experimental/vla/models/openvla_oft/configuration_prismatic` | `verl/experimental/vla/models/openvla_oft/configuration_prismatic.py` | 0 | transformers, transformers.models.auto, typing |
| `verl/experimental/vla/models/openvla_oft/constants` | `verl/experimental/vla/models/openvla_oft/constants.py` | 0 | enum, sys |
| `verl/experimental/vla/models/openvla_oft/modeling_prismatic` | `verl/experimental/vla/models/openvla_oft/modeling_prismatic.py` | 3 | dataclasses, functools, logging, numpy, timm |
| `verl/experimental/vla/models/openvla_oft/processing_prismatic` | `verl/experimental/vla/models/openvla_oft/processing_prismatic.py` | 1 | PIL, timm.data, torch, torchvision.transforms, torchvision.transforms.functional |
| `verl/experimental/vla/models/openvla_oft/train_utils` | `verl/experimental/vla/models/openvla_oft/train_utils.py` | 1 | torch |
| `verl/experimental/vla/models/pi0_torch/__init__` | `verl/experimental/vla/models/pi0_torch/__init__.py` | 2 | - |
| `verl/experimental/vla/models/pi0_torch/configuration_pi0_torch` | `verl/experimental/vla/models/pi0_torch/configuration_pi0_torch.py` | 0 | transformers |
| `verl/experimental/vla/models/pi0_torch/model/modeling_pi0` | `verl/experimental/vla/models/pi0_torch/model/modeling_pi0.py` | 1 | diffusers.configuration_utils, diffusers.models.modeling_utils, math, torch, torch.nn.functional |
| `verl/experimental/vla/models/pi0_torch/model/paligemma_with_expert` | `verl/experimental/vla/models/pi0_torch/model/paligemma_with_expert.py` | 2 | torch, torch.nn.functional, transformers.activations, transformers.modeling_outputs, transformers.models.auto |
| `verl/experimental/vla/models/pi0_torch/modeling_pi0_torch` | `verl/experimental/vla/models/pi0_torch/modeling_pi0_torch.py` | 8 | __future__, math, onnx_ir, torch, torch.distributed.fsdp |
| `verl/experimental/vla/models/pi0_torch/pi0_utils` | `verl/experimental/vla/models/pi0_torch/pi0_utils.py` | 0 | torch, torch.nn.functional, torchvision, typing |
| `verl/experimental/vla/models/pi0_torch/policy/__init__` | `verl/experimental/vla/models/pi0_torch/policy/__init__.py` | 0 | - |
| `verl/experimental/vla/models/pi0_torch/policy/base` | `verl/experimental/vla/models/pi0_torch/policy/base.py` | 0 | abc, torch |
| `verl/experimental/vla/models/pi0_torch/policy/libero_policy` | `verl/experimental/vla/models/pi0_torch/policy/libero_policy.py` | 2 | torch, typing_extensions |
| `verl/experimental/vla/models/register_vla_models` | `verl/experimental/vla/models/register_vla_models.py` | 4 | .pi0_torch, transformers |
| `verl/experimental/vla/naive_rollout_rob` | `verl/experimental/vla/naive_rollout_rob.py` | 5 | PIL, json, logging, os, torch |
| `verl/experimental/vla/prepare_libero_dataset` | `verl/experimental/vla/prepare_libero_dataset.py` | 0 | argparse, datasets, libero.libero, libero.libero.benchmark, numpy |
| `verl/experimental/vla/rob_ray_trainer` | `verl/experimental/vla/rob_ray_trainer.py` | 12 | asyncio, collections, numpy, omegaconf, pprint |
| `verl/experimental/vla/sac/base` | `verl/experimental/vla/sac/base.py` | 0 | abc, torch, typing, verl |
| `verl/experimental/vla/sac/naive_rollout_pi05` | `verl/experimental/vla/sac/naive_rollout_pi05.py` | 2 | logging, torch, torch.distributed.fsdp, typing, verl |
| `verl/experimental/vla/sac/replay_pool` | `verl/experimental/vla/sac/replay_pool.py` | 0 | dataclasses, logging, os, tensordict, torch |
| `verl/experimental/vla/sac/sac_actor` | `verl/experimental/vla/sac/sac_actor.py` | 4 | logging, numpy, os, tensordict, torch |
| `verl/experimental/vla/sac/sac_ray_trainer` | `verl/experimental/vla/sac/sac_ray_trainer.py` | 6 | asyncio, numpy, omegaconf, pprint, torch |
| `verl/experimental/vla/workers/env/env_loop_wg_test` | `verl/experimental/vla/workers/env/env_loop_wg_test.py` | 3 | asyncio, omegaconf, ray, verl |
| `verl/experimental/vla/workers/env/env_manager` | `verl/experimental/vla/workers/env/env_manager.py` | 1 | gc, logging, os, pynvml, subprocess |
| `verl/experimental/vla/workers/env/env_worker` | `verl/experimental/vla/workers/env/env_worker.py` | 8 | itertools, omegaconf, torch, torch.distributed.device_mesh, verl |
| `verl/interactions/__init__` | `verl/interactions/__init__.py` | 0 | - |
| `verl/interactions/base` | `verl/interactions/base.py` | 0 | typing, uuid |
| `verl/interactions/gsm8k_interaction` | `verl/interactions/gsm8k_interaction.py` | 1 | logging, os, typing, uuid, verl.utils.reward_score |
| `verl/interactions/gym_env` | `verl/interactions/gym_env.py` | 0 | __future__, abc, aiohttp, typing |
| `verl/interactions/gym_interaction` | `verl/interactions/gym_interaction.py` | 2 | __future__, copy, typing, uuid |
| `verl/interactions/utils/__init__` | `verl/interactions/utils/__init__.py` | 0 | - |
| `verl/interactions/utils/interaction_registry` | `verl/interactions/utils/interaction_registry.py` | 1 | logging, omegaconf, os, sys |
| `verl/interactions/weather_interaction` | `verl/interactions/weather_interaction.py` | 1 | logging, os, typing, uuid |
| `verl/model_merger/__init__` | `verl/model_merger/__init__.py` | 0 | - |
| `verl/model_merger/__main__` | `verl/model_merger/__main__.py` | 3 | - |
| `verl/model_merger/base_model_merger` | `verl/model_merger/base_model_merger.py` | 2 | abc, accelerate, argparse, dataclasses, huggingface_hub |
| `verl/model_merger/fsdp_model_merger` | `verl/model_merger/fsdp_model_merger.py` | 1 | concurrent.futures, json, numpy, os, pathlib |
| `verl/model_merger/megatron_model_merger` | `verl/model_merger/megatron_model_merger.py` | 7 | accelerate, contextlib, json, megatron.core.models.gpt.gpt_model, megatron.core.tensor_parallel.random |
| `verl/models/__init__` | `verl/models/__init__.py` | 0 | - |
| `verl/models/mcore/__init__` | `verl/models/mcore/__init__.py` | 2 | - |
| `verl/models/mcore/bridge` | `verl/models/mcore/bridge.py` | 2 | megatron.bridge.models.conversion.param_mapping, megatron.bridge.peft.canonical_lora, megatron.bridge.peft.dora, megatron.bridge.peft.lora, torch |
| `verl/models/mcore/config_converter` | `verl/models/mcore/config_converter.py` | 3 | megatron.core.transformer, megatron.core.transformer.enums, torch, torch.nn.functional, transformers |
| `verl/models/mcore/loader` | `verl/models/mcore/loader.py` | 5 | megatron.core.transformer.module, time, torch, torch.nn.parallel, verl.utils.logger |
| `verl/models/mcore/mbridge` | `verl/models/mcore/mbridge.py` | 2 | mbridge.utils.post_creation_callbacks |
| `verl/models/mcore/model_forward` | `verl/models/mcore/model_forward.py` | 3 | torch, torch.nested._internal.nested_tensor |
| `verl/models/mcore/model_forward_1f1b_overlap` | `verl/models/mcore/model_forward_1f1b_overlap.py` | 7 | megatron.core.models.common.model_chunk_schedule_plan, megatron.core.models.gpt.gpt_model, megatron.core.transformer.multi_token_prediction, torch, typing |
| `verl/models/mcore/model_forward_fused` | `verl/models/mcore/model_forward_fused.py` | 6 | collections, megatron.core.config_logger, megatron.core.inference.contexts, megatron.core.models.gpt.gpt_model, megatron.core.packed_seq_params |
| `verl/models/mcore/model_initializer` | `verl/models/mcore/model_initializer.py` | 2 | .qwen2_5_vl, abc, copy, inspect, megatron.core.extensions.transformer_engine |
| `verl/models/mcore/mtp_patch` | `verl/models/mcore/mtp_patch.py` | 3 | megatron.core.models.gpt.gpt_model, megatron.core.transformer.multi_token_prediction, torch, typing |
| `verl/models/mcore/patch` | `verl/models/mcore/patch.py` | 2 | inspect, logging, megatron.core.dist_checkpointing.strategies.async_utils, megatron.core.dist_checkpointing.strategies.filesystem_async, megatron.core.tensor_parallel.random |
| `verl/models/mcore/qwen2_5_vl/__init__` | `verl/models/mcore/qwen2_5_vl/__init__.py` | 2 | - |
| `verl/models/mcore/qwen2_5_vl/attention` | `verl/models/mcore/qwen2_5_vl/attention.py` | 2 | - |
| `verl/models/mcore/qwen2_5_vl/model` | `verl/models/mcore/qwen2_5_vl/model.py` | 5 | logging, megatron.core.models.gpt.gpt_model, megatron.core.packed_seq_params, megatron.core.transformer, megatron.core.transformer.spec_utils |
| `verl/models/mcore/qwen2_5_vl/rope_utils` | `verl/models/mcore/qwen2_5_vl/rope_utils.py` | 1 | __future__, logging, torch, typing |
| `verl/models/mcore/qwen2_5_vl/vision_config` | `verl/models/mcore/qwen2_5_vl/vision_config.py` | 1 | megatron.core.transformer, torch |
| `verl/models/mcore/qwen2_5_vl/vision_model` | `verl/models/mcore/qwen2_5_vl/vision_model.py` | 2 | megatron.core.models.common.vision_module.vision_module, megatron.core.models.vision.multimodal_projector, megatron.core.packed_seq_params, megatron.core.transformer.enums, megatron.core.transformer.spec_utils |
| `verl/models/mcore/qwen2_5_vl/vision_transformer_block` | `verl/models/mcore/qwen2_5_vl/vision_transformer_block.py` | 0 | megatron.core.transformer.transformer_block |
| `verl/models/mcore/registry` | `verl/models/mcore/registry.py` | 5 | enum, torch, torch.nn, typing |
| `verl/models/mcore/saver` | `verl/models/mcore/saver.py` | 4 | megatron.core.transformer.module, time, torch, torch.nn.parallel, verl.utils.logger |
| `verl/models/mcore/util` | `verl/models/mcore/util.py` | 3 | logging, math, megatron.core.packed_seq_params, os, torch |
| `verl/models/mcore/weight_converter` | `verl/models/mcore/weight_converter.py` | 0 | megatron.core.transformer, torch, transformers |
| `verl/models/registry` | `verl/models/registry.py` | 0 | importlib, torch.nn, typing |
| `verl/models/transformers/__init__` | `verl/models/transformers/__init__.py` | 2 | - |
| `verl/models/transformers/apertus` | `verl/models/transformers/apertus.py` | 2 | sys, torch, transformers.cache_utils, transformers.modeling_utils, transformers.models.apertus.modeling_apertus |
| `verl/models/transformers/dense_common` | `verl/models/transformers/dense_common.py` | 2 | dataclasses, torch, transformers.cache_utils, transformers.modeling_outputs, typing |
| `verl/models/transformers/glm4v` | `verl/models/transformers/glm4v.py` | 6 | dataclasses, flash_attn, inspect, itertools, logging |
| `verl/models/transformers/kimi_vl` | `verl/models/transformers/kimi_vl.py` | 3 | torch, torch.nn.functional, transformers.cache_utils, transformers.modeling_flash_attention_utils, typing |
| `verl/models/transformers/llama` | `verl/models/transformers/llama.py` | 3 | sys, torch, transformers.cache_utils, transformers.modeling_flash_attention_utils, transformers.modeling_utils |
| `verl/models/transformers/monkey_patch` | `verl/models/transformers/monkey_patch.py` | 9 | sys, torch, transformers.integrations, transformers.modeling_flash_attention_utils, transformers.modeling_utils |
| `verl/models/transformers/npu_patch` | `verl/models/transformers/npu_patch.py` | 3 | torch, torch.nn.functional, torch_npu, transformers.activations, transformers.models.qwen2_5_vl |
| `verl/models/transformers/qwen2` | `verl/models/transformers/qwen2.py` | 3 | torch, transformers.cache_utils, transformers.modeling_flash_attention_utils, transformers.modeling_utils, transformers.models.llama.modeling_llama |
| `verl/models/transformers/qwen2_vl` | `verl/models/transformers/qwen2_vl.py` | 7 | dataclasses, flash_attn, inspect, logging, os |
| `verl/models/transformers/qwen3_vl` | `verl/models/transformers/qwen3_vl.py` | 2 | dataclasses, functools, logging, os, torch |
| `verl/models/transformers/tiled_mlp` | `verl/models/transformers/tiled_mlp.py` | 0 | importlib, threading, torch, torch.nn, typing |
| `verl/models/weight_loader_registry` | `verl/models/weight_loader_registry.py` | 2 | - |
| `verl/protocol` | `verl/protocol.py` | 6 | contextlib, copy, dataclasses, functools, io |
| `verl/single_controller/__init__` | `verl/single_controller/__init__.py` | 1 | ., os |
| `verl/single_controller/base/__init__` | `verl/single_controller/base/__init__.py` | 2 | - |
| `verl/single_controller/base/decorator` | `verl/single_controller/base/decorator.py` | 4 | functools, inspect, os, types |
| `verl/single_controller/base/worker` | `verl/single_controller/base/worker.py` | 3 | dataclasses, os, ray, socket, warnings |
| `verl/single_controller/base/worker_group` | `verl/single_controller/base/worker_group.py` | 1 | logging, signal, threading, time, typing |
| `verl/single_controller/ray/__init__` | `verl/single_controller/ray/__init__.py` | 1 | - |
| `verl/single_controller/ray/base` | `verl/single_controller/ray/base.py` | 5 | copy, dataclasses, inspect, logging, numpy |
| `verl/third_party/__init__` | `verl/third_party/__init__.py` | 0 | - |
| `verl/third_party/torch/__init__` | `verl/third_party/torch/__init__.py` | 0 | - |
| `verl/third_party/torch/distributed/__init__` | `verl/third_party/torch/distributed/__init__.py` | 0 | - |
| `verl/third_party/torch/distributed/_state_dict_utils` | `verl/third_party/torch/distributed/_state_dict_utils.py` | 1 | collections.abc, copy, io, math, torch |
| `verl/third_party/torch/distributed/checkpoint/__init__` | `verl/third_party/torch/distributed/checkpoint/__init__.py` | 0 | - |
| `verl/third_party/torch/distributed/checkpoint/state_dict` | `verl/third_party/torch/distributed/checkpoint/state_dict.py` | 3 | collections.abc, contextlib, dataclasses, functools, gc |
| `verl/third_party/vllm/__init__` | `verl/third_party/vllm/__init__.py` | 3 | importlib.metadata, packaging, vllm |
| `verl/tools/__init__` | `verl/tools/__init__.py` | 0 | - |
| `verl/tools/base_tool` | `verl/tools/base_tool.py` | 2 | json, typing, uuid |
| `verl/tools/geo3k_tool` | `verl/tools/geo3k_tool.py` | 3 | logging, os, typing, uuid, verl.utils.reward_score |
| `verl/tools/gsm8k_tool` | `verl/tools/gsm8k_tool.py` | 3 | logging, os, typing, uuid, verl.utils.reward_score |
| `verl/tools/image_zoom_in_tool` | `verl/tools/image_zoom_in_tool.py` | 3 | contextlib, enum, logging, math, os |
| `verl/tools/mcp_base_tool` | `verl/tools/mcp_base_tool.py` | 4 | fastmcp.exceptions, json, logging, os, typing |
| `verl/tools/mcp_search_tool` | `verl/tools/mcp_search_tool.py` | 2 | json, logging, os, re |
| `verl/tools/sandbox_fusion_tools` | `verl/tools/sandbox_fusion_tools.py` | 4 | contextlib, enum, logging, os, ray |
| `verl/tools/schemas` | `verl/tools/schemas.py` | 0 | json, pydantic, typing |
| `verl/tools/search_tool` | `verl/tools/search_tool.py` | 5 | contextlib, enum, json, logging, os |
| `verl/tools/utils/__init__` | `verl/tools/utils/__init__.py` | 0 | - |
| `verl/tools/utils/mcp_clients/McpClientManager` | `verl/tools/utils/mcp_clients/McpClientManager.py` | 1 | asyncio, fastmcp, fastmcp.client.transports, json, logging |
| `verl/tools/utils/mcp_clients/utils` | `verl/tools/utils/mcp_clients/utils.py` | 0 | logging, mcp, threading, time |
| `verl/tools/utils/search_r1_like_utils` | `verl/tools/utils/search_r1_like_utils.py` | 0 | json, logging, requests, threading, time |
| `verl/tools/utils/tool_registry` | `verl/tools/utils/tool_registry.py` | 2 | asyncio, enum, importlib, logging, omegaconf |
| `verl/trainer/__init__` | `verl/trainer/__init__.py` | 0 | - |
| `verl/trainer/config/__init__` | `verl/trainer/config/__init__.py` | 2 | . |
| `verl/trainer/config/algorithm` | `verl/trainer/config/algorithm.py` | 1 | dataclasses, typing |
| `verl/trainer/config/config` | `verl/trainer/config/config.py` | 1 | dataclasses, typing |
| `verl/trainer/constants_ppo` | `verl/trainer/constants_ppo.py` | 1 | json, os |
| `verl/trainer/distillation/__init__` | `verl/trainer/distillation/__init__.py` | 1 | - |
| `verl/trainer/distillation/fsdp/losses` | `verl/trainer/distillation/fsdp/losses.py` | 2 | torch, torch.nn.functional |
| `verl/trainer/distillation/losses` | `verl/trainer/distillation/losses.py` | 7 | dataclasses, tensordict, torch, typing, verl.utils.metric |
| `verl/trainer/distillation/megatron/losses` | `verl/trainer/distillation/megatron/losses.py` | 3 | megatron.core.fusions.fused_cross_entropy, megatron.core.parallel_state, torch, typing |
| `verl/trainer/main_eval` | `verl/trainer/main_eval.py` | 2 | collections, hydra, numpy, omegaconf, pandas |
| `verl/trainer/main_generation_server` | `verl/trainer/main_generation_server.py` | 2 | aiohttp, asyncio, hydra, itertools, numpy |
| `verl/trainer/main_ppo` | `verl/trainer/main_ppo.py` | 15 | hydra, omegaconf, os, pprint, ray |
| `verl/trainer/ppo/__init__` | `verl/trainer/ppo/__init__.py` | 0 | - |
| `verl/trainer/ppo/core_algos` | `verl/trainer/ppo/core_algos.py` | 5 | collections, copy, enum, numpy, omegaconf |
| `verl/trainer/ppo/metric_utils` | `verl/trainer/ppo/metric_utils.py` | 2 | collections, functools, numpy, torch, typing |
| `verl/trainer/ppo/prefix_grouper_utils` | `verl/trainer/ppo/prefix_grouper_utils.py` | 1 | __future__, prefix_grouper, torch |
| `verl/trainer/ppo/ray_trainer` | `verl/trainer/ppo/ray_trainer.py` | 28 | collections, copy, functools, json, numpy |
| `verl/trainer/ppo/reward` | `verl/trainer/ppo/reward.py` | 4 | __future__, functools, inspect, multiprocessing, omegaconf |
| `verl/trainer/ppo/rollout_corr_helper` | `verl/trainer/ppo/rollout_corr_helper.py` | 4 | math, omegaconf, torch, typing |
| `verl/trainer/ppo/utils` | `verl/trainer/ppo/utils.py` | 3 | enum, omegaconf, warnings |
| `verl/trainer/sft_trainer` | `verl/trainer/sft_trainer.py` | 11 | functools, hydra, logging, omegaconf, os |
| `verl/trainer/sft_trainer_ray` | `verl/trainer/sft_trainer_ray.py` | 10 | functools, hydra, logging, omegaconf, os |
| `verl/utils/__init__` | `verl/utils/__init__.py` | 3 | . |
| `verl/utils/activation_offload` | `verl/utils/activation_offload.py` | 2 | __future__, functools, logging, os, torch |
| `verl/utils/attention_utils` | `verl/utils/attention_utils.py` | 2 | flash_attn.bert_padding, typing |
| `verl/utils/chat_template` | `verl/utils/chat_template.py` | 1 | logging, os, transformers |
| `verl/utils/checkpoint/__init__` | `verl/utils/checkpoint/__init__.py` | 1 | - |
| `verl/utils/checkpoint/checkpoint_handler` | `verl/utils/checkpoint/checkpoint_handler.py` | 4 | enum, json, logging, os, re |
| `verl/utils/checkpoint/checkpoint_manager` | `verl/utils/checkpoint/checkpoint_manager.py` | 3 | datetime, numpy, omegaconf, os, random |
| `verl/utils/checkpoint/fsdp_checkpoint_manager` | `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | 6 | accelerate, dataclasses, json, logging, omegaconf |
| `verl/utils/checkpoint/megatron_checkpoint_manager` | `verl/utils/checkpoint/megatron_checkpoint_manager.py` | 10 | accelerate, collections.abc, dataclasses, inspect, json |
| `verl/utils/config` | `verl/utils/config.py` | 2 | dataclasses, omegaconf, typing |
| `verl/utils/dataset/__init__` | `verl/utils/dataset/__init__.py` | 2 | - |
| `verl/utils/dataset/dataset_utils` | `verl/utils/dataset/dataset_utils.py` | 0 | enum, tensordict.tensorclass, torch, torch.utils.data |
| `verl/utils/dataset/multiturn_sft_dataset` | `verl/utils/dataset/multiturn_sft_dataset.py` | 7 | functools, logging, numpy, omegaconf, os |
| `verl/utils/dataset/rl_dataset` | `verl/utils/dataset/rl_dataset.py` | 5 | PIL, collections, copy, datasets, io |
| `verl/utils/dataset/rm_dataset` | `verl/utils/dataset/rm_dataset.py` | 3 | numpy, os, pandas, torch, torch.utils.data |
| `verl/utils/dataset/vision_utils` | `verl/utils/dataset/vision_utils.py` | 0 | PIL, io, qwen_vl_utils, torch, typing |
| `verl/utils/debug/__init__` | `verl/utils/debug/__init__.py` | 0 | ..profiler |
| `verl/utils/debug/metrics` | `verl/utils/debug/metrics.py` | 1 | logging, torch |
| `verl/utils/debug/performance` | `verl/utils/debug/performance.py` | 1 | - |
| `verl/utils/debug/trajectory_tracker` | `verl/utils/debug/trajectory_tracker.py` | 1 | collections, io, os, ray, tempfile |
| `verl/utils/device` | `verl/utils/device.py` | 0 | logging, os, packaging, platform, re |
| `verl/utils/distributed` | `verl/utils/distributed.py` | 4 | ctypes, datetime, os, pynvml, ray |
| `verl/utils/experimental/__init__` | `verl/utils/experimental/__init__.py` | 0 | - |
| `verl/utils/experimental/reward_utils` | `verl/utils/experimental/reward_utils.py` | 0 | PIL, base64, io |
| `verl/utils/experimental/torch_functional` | `verl/utils/experimental/torch_functional.py` | 0 | flash_attn.ops.triton.cross_entropy, torch, typing |
| `verl/utils/flops_counter` | `verl/utils/flops_counter.py` | 1 | inspect, torch, transformers |
| `verl/utils/fp8_utils` | `verl/utils/fp8_utils.py` | 2 | logging, os, torch |
| `verl/utils/fs` | `verl/utils/fs.py` | 1 | filelock, hashlib, huggingface_hub, os, shutil |
| `verl/utils/fsdp_utils` | `verl/utils/fsdp_utils.py` | 4 | abc, accelerate, collections, contextlib, functools |
| `verl/utils/groupwise` | `verl/utils/groupwise.py` | 1 | __future__, numpy, os, torch, typing |
| `verl/utils/hdfs_io` | `verl/utils/hdfs_io.py` | 0 | logging, os, shutil |
| `verl/utils/import_utils` | `verl/utils/import_utils.py` | 1 | functools, importlib, os, sys, typing |
| `verl/utils/kernel/__init__` | `verl/utils/kernel/__init__.py` | 0 | - |
| `verl/utils/kernel/fp8_kernel` | `verl/utils/kernel/fp8_kernel.py` | 0 | logging, os, torch, triton, triton.language |
| `verl/utils/kernel/kernels` | `verl/utils/kernel/kernels.py` | 2 | contextlib, contextvars, dataclasses, torch, triton |
| `verl/utils/kernel/linear_cross_entropy` | `verl/utils/kernel/linear_cross_entropy.py` | 1 | ., torch, typing |
| `verl/utils/logger/__init__` | `verl/utils/logger/__init__.py` | 1 | - |
| `verl/utils/logger/aggregate_logger` | `verl/utils/logger/aggregate_logger.py` | 0 | datetime, logging, numbers, pprint, torch |
| `verl/utils/logging_utils` | `verl/utils/logging_utils.py` | 0 | logging, os, torch |
| `verl/utils/megatron/__init__` | `verl/utils/megatron/__init__.py` | 0 | - |
| `verl/utils/megatron/dist_checkpointing` | `verl/utils/megatron/dist_checkpointing.py` | 1 | megatron.core.dist_checkpointing.serialization, megatron.core.dist_checkpointing.strategies.fully_parallel, packaging, torch, transformer_engine |
| `verl/utils/megatron/memory` | `verl/utils/megatron/memory.py` | 1 | torch |
| `verl/utils/megatron/optimizer` | `verl/utils/megatron/optimizer.py` | 1 | megatron.core.optimizer_param_scheduler, torch, verl.utils.logger |
| `verl/utils/megatron/pipeline_parallel` | `verl/utils/megatron/pipeline_parallel.py` | 2 | flash_attn.bert_padding, torch |
| `verl/utils/megatron/router_replay_patch` | `verl/utils/megatron/router_replay_patch.py` | 0 | enum, functools, inspect, megatron.core.transformer.moe.moe_utils, megatron.core.transformer.moe.router |
| `verl/utils/megatron/router_replay_utils` | `verl/utils/megatron/router_replay_utils.py` | 6 | inspect, megatron.core.pipeline_parallel.schedules, megatron.core.transformer.enums, megatron.core.transformer.transformer_config, megatron.core.transformer.transformer_layer |
| `verl/utils/megatron/sequence_parallel` | `verl/utils/megatron/sequence_parallel.py` | 1 | torch, torch.nn.functional |
| `verl/utils/megatron/tensor_parallel` | `verl/utils/megatron/tensor_parallel.py` | 2 | flash_attn.bert_padding, torch, torch.nn, typing |
| `verl/utils/megatron_peft_utils` | `verl/utils/megatron_peft_utils.py` | 1 | peft, torch, typing |
| `verl/utils/megatron_utils` | `verl/utils/megatron_utils.py` | 18 | dataclasses, gc, inspect, logging, megatron.bridge.training.checkpointing |
| `verl/utils/memory_utils` | `verl/utils/memory_utils.py` | 1 | datetime, gc, inspect, logging, os |
| `verl/utils/metric/__init__` | `verl/utils/metric/__init__.py` | 1 | - |
| `verl/utils/metric/utils` | `verl/utils/metric/utils.py` | 0 | enum, numpy, torch, typing |
| `verl/utils/model` | `verl/utils/model.py` | 11 | accelerate, dataclasses, json, megatron.core.dist_checkpointing.serialization, megatron.core.models.gpt.gpt_layer_specs |
| `verl/utils/modelopt/__init__` | `verl/utils/modelopt/__init__.py` | 5 | - |
| `verl/utils/modelopt/megatron_qat_patch` | `verl/utils/modelopt/megatron_qat_patch.py` | 4 | gc, megatron.bridge.models.conversion.model_bridge, megatron.bridge.models.conversion.param_mapping, megatron.core.dist_checkpointing.mapping, re |
| `verl/utils/modelopt/qat_utils` | `verl/utils/modelopt/qat_utils.py` | 3 | megatron.bridge.models.conversion.param_mapping, megatron.bridge.models.gpt_provider |
| `verl/utils/modelopt/qat_weight_exporter` | `verl/utils/modelopt/qat_weight_exporter.py` | 2 | dataclasses, megatron.bridge.models.conversion.model_bridge, modelopt.torch.export.quant_utils, modelopt.torch.quantization.qtensor.nvfp4_tensor, re |
| `verl/utils/modelopt/quantize` | `verl/utils/modelopt/quantize.py` | 1 | modelopt.torch.quantization, torch.nn |
| `verl/utils/modelopt/vllm_modelopt_patch` | `verl/utils/modelopt/vllm_modelopt_patch.py` | 1 | torch, torch.nn, typing, vllm._custom_ops, vllm.model_executor.layers.quantization.kv_cache |
| `verl/utils/net_utils` | `verl/utils/net_utils.py` | 0 | ipaddress, socket |
| `verl/utils/npu_flash_attn_utils` | `verl/utils/npu_flash_attn_utils.py` | 0 | einops, torch, torch.nn.functional |
| `verl/utils/profiler/__init__` | `verl/utils/profiler/__init__.py` | 7 | - |
| `verl/utils/profiler/config` | `verl/utils/profiler/config.py` | 1 | dataclasses, json, omegaconf, os, typing |
| `verl/utils/profiler/empty_annotations` | `verl/utils/profiler/empty_annotations.py` | 0 | typing |
| `verl/utils/profiler/mstx_profile` | `verl/utils/profiler/mstx_profile.py` | 3 | contextlib, functools, logging, os, packaging |
| `verl/utils/profiler/nvtx_profile` | `verl/utils/profiler/nvtx_profile.py` | 3 | contextlib, functools, nvtx, torch, typing |
| `verl/utils/profiler/performance` | `verl/utils/profiler/performance.py` | 2 | codetiming, contextlib, datetime, inspect, logging |
| `verl/utils/profiler/profile` | `verl/utils/profiler/profile.py` | 6 | functools, typing |
| `verl/utils/profiler/torch_profile` | `verl/utils/profiler/torch_profile.py` | 2 | datetime, functools, os, torch, typing |
| `verl/utils/py_functional` | `verl/utils/py_functional.py` | 0 | contextlib, functools, importlib, multiprocessing, numpy |
| `verl/utils/qat/__init__` | `verl/utils/qat/__init__.py` | 2 | verl.utils.qat |
| `verl/utils/qat/core` | `verl/utils/qat/core.py` | 2 | dataclasses, json, logging, re, torch.nn |
| `verl/utils/qat/linear` | `verl/utils/qat/linear.py` | 0 | enum, torch, torch.nn, torch.nn.functional, triton |
| `verl/utils/qat/quantizer` | `verl/utils/qat/quantizer.py` | 1 | compressed_tensors.compressors.quantized_compressors.fp4_quantized, compressed_tensors.quantization.quant_args, compressed_tensors.quantization.utils.helpers, logging, os |
| `verl/utils/qat/vllm_patch` | `verl/utils/qat/vllm_patch.py` | 2 | flashinfer, logging, os, torch, torch.nn |
| `verl/utils/ray_utils` | `verl/utils/ray_utils.py` | 0 | asyncio, concurrent.futures, functools, inspect, os |
| `verl/utils/rendezvous/__init__` | `verl/utils/rendezvous/__init__.py` | 0 | - |
| `verl/utils/rendezvous/ray_backend` | `verl/utils/rendezvous/ray_backend.py` | 1 | cupy.cuda.nccl, logging, ray, time |
| `verl/utils/reward_score/__init__` | `verl/utils/reward_score/__init__.py` | 1 | . |
| `verl/utils/reward_score/geo3k` | `verl/utils/reward_score/geo3k.py` | 1 | re |
| `verl/utils/reward_score/gsm8k` | `verl/utils/reward_score/gsm8k.py` | 0 | re |
| `verl/utils/reward_score/jpeg_compressibility` | `verl/utils/reward_score/jpeg_compressibility.py` | 0 | PIL, io, numpy, torch |
| `verl/utils/reward_score/math_batch` | `verl/utils/reward_score/math_batch.py` | 1 | - |
| `verl/utils/reward_score/math_dapo` | `verl/utils/reward_score/math_dapo.py` | 0 | re, typing |
| `verl/utils/reward_score/math_reward` | `verl/utils/reward_score/math_reward.py` | 0 | - |
| `verl/utils/reward_score/math_verify` | `verl/utils/reward_score/math_verify.py` | 0 | math_verify.errors, math_verify.metric, math_verify.parser |
| `verl/utils/reward_score/prime_code/__init__` | `verl/utils/reward_score/prime_code/__init__.py` | 1 | json, traceback |
| `verl/utils/reward_score/prime_code/testing_util` | `verl/utils/reward_score/prime_code/testing_util.py` | 0 | ast, builtins, datetime, enum, faulthandler |
| `verl/utils/reward_score/prime_code/utils` | `verl/utils/reward_score/prime_code/utils.py` | 1 | multiprocessing, os, sys, traceback, typing |
| `verl/utils/reward_score/prime_math/__init__` | `verl/utils/reward_score/prime_math/__init__.py` | 2 | ., contextlib, math, pylatexenc, re |
| `verl/utils/reward_score/prime_math/grader` | `verl/utils/reward_score/prime_math/grader.py` | 1 | contextlib, math, re, sympy, sympy.parsing.latex |
| `verl/utils/reward_score/prime_math/math_normalize` | `verl/utils/reward_score/prime_math/math_normalize.py` | 0 | re, typing |
| `verl/utils/reward_score/rlla` | `verl/utils/reward_score/rlla.py` | 0 | collections, json, random, re |
| `verl/utils/reward_score/sandbox_fusion/__init__` | `verl/utils/reward_score/sandbox_fusion/__init__.py` | 1 | json, logging, traceback |
| `verl/utils/reward_score/sandbox_fusion/utils` | `verl/utils/reward_score/sandbox_fusion/utils.py` | 0 | bisect, builtins, collections, concurrent.futures, copy |
| `verl/utils/reward_score/search_r1_like_qa_em` | `verl/utils/reward_score/search_r1_like_qa_em.py` | 0 | random, re, string |
| `verl/utils/rollout_skip` | `verl/utils/rollout_skip.py` | 3 | enum, json, pathlib, typing |
| `verl/utils/rollout_trace` | `verl/utils/rollout_trace.py` | 1 | contextlib, contextvars, functools, inspect, mlflow |
| `verl/utils/seqlen_balancing` | `verl/utils/seqlen_balancing.py` | 3 | copy, heapq, itertools, torch |
| `verl/utils/sglang/sglang_fp8_utils` | `verl/utils/sglang/sglang_fp8_utils.py` | 1 | - |
| `verl/utils/tensordict_utils` | `verl/utils/tensordict_utils.py` | 0 | logging, tensordict, tensordict.tensorclass, torch, torch.utils.data |
| `verl/utils/tokenizer` | `verl/utils/tokenizer.py` | 3 | transformers, transformers.models.qwen2_5_vl, types, warnings |
| `verl/utils/torch_dtypes` | `verl/utils/torch_dtypes.py` | 0 | torch |
| `verl/utils/torch_functional` | `verl/utils/torch_functional.py` | 3 | contextlib, flash_attn.bert_padding, flash_attn.ops.triton.cross_entropy, math, mindspeed.patch_utils |
| `verl/utils/tracking` | `verl/utils/tracking.py` | 0 | clearml, dataclasses, enum, functools, json |
| `verl/utils/transformers_compat` | `verl/utils/transformers_compat.py` | 0 | functools, importlib.metadata, packaging, transformers, transformers.modeling_flash_attention_utils |
| `verl/utils/trtllm/trtllm_fp8_utils` | `verl/utils/trtllm/trtllm_fp8_utils.py` | 1 | - |
| `verl/utils/ulysses` | `verl/utils/ulysses.py` | 1 | torch, typing |
| `verl/utils/vllm/__init__` | `verl/utils/vllm/__init__.py` | 2 | - |
| `verl/utils/vllm/npu_vllm_patch` | `verl/utils/vllm/npu_vllm_patch.py` | 2 | functools, os, packaging, torch_npu, vllm |
| `verl/utils/vllm/patch` | `verl/utils/vllm/patch.py` | 1 | vllm.model_executor.models.deepseek_v2, vllm.model_executor.models.mixtral, vllm.model_executor.models.qwen2_moe, vllm.model_executor.models.qwen3_5, vllm.model_executor.models.qwen3_moe |
| `verl/utils/vllm/utils` | `verl/utils/vllm/utils.py` | 1 | msgspec, packaging, verl.third_party.vllm, vllm.lora.lora_model, vllm.lora.models |
| `verl/utils/vllm/vllm_fp8_utils` | `verl/utils/vllm/vllm_fp8_utils.py` | 3 | dataclasses, inspect, logging, packaging, torch |
| `verl/utils/vllm_omni/__init__` | `verl/utils/vllm_omni/__init__.py` | 1 | - |
| `verl/utils/vllm_omni/utils` | `verl/utils/vllm_omni/utils.py` | 1 | msgspec, vllm.lora.lora_model, vllm.lora.models, vllm.lora.peft_helper, vllm_omni.diffusion.lora.manager |
| `verl/workers/__init__` | `verl/workers/__init__.py` | 0 | - |
| `verl/workers/actor/__init__` | `verl/workers/actor/__init__.py` | 2 | - |
| `verl/workers/actor/base` | `verl/workers/actor/base.py` | 0 | abc, torch, verl |
| `verl/workers/actor/dp_actor` | `verl/workers/actor/dp_actor.py` | 16 | logging, os, torch, torch.distributed.fsdp, torch.distributed.fsdp.sharded_grad_scaler |
| `verl/workers/actor/megatron_actor` | `verl/workers/actor/megatron_actor.py` | 21 | functools, itertools, logging, megatron.core.models.gpt.gpt_model, omegaconf |
| `verl/workers/config/__init__` | `verl/workers/config/__init__.py` | 8 | . |
| `verl/workers/config/actor` | `verl/workers/config/actor.py` | 6 | dataclasses, omegaconf, typing, verl.utils.qat |
| `verl/workers/config/critic` | `verl/workers/config/critic.py` | 5 | dataclasses, omegaconf, typing, verl.utils.profiler, warnings |
| `verl/workers/config/distillation` | `verl/workers/config/distillation.py` | 3 | dataclasses, logging, os, typing |
| `verl/workers/config/engine` | `verl/workers/config/engine.py` | 4 | ...utils.profiler, dataclasses, typing, warnings |
| `verl/workers/config/megatron_peft` | `verl/workers/config/megatron_peft.py` | 2 | - |
| `verl/workers/config/model` | `verl/workers/config/model.py` | 5 | dataclasses, json, omegaconf, os, transformers |
| `verl/workers/config/optimizer` | `verl/workers/config/optimizer.py` | 1 | dataclasses, importlib, omegaconf, typing, warnings |
| `verl/workers/config/reward` | `verl/workers/config/reward.py` | 4 | dataclasses, logging, os, typing |
| `verl/workers/config/rollout` | `verl/workers/config/rollout.py` | 2 | dataclasses, omegaconf, typing, verl.utils.profiler, warnings |
| `verl/workers/critic/__init__` | `verl/workers/critic/__init__.py` | 2 | - |
| `verl/workers/critic/base` | `verl/workers/critic/base.py` | 0 | abc, torch, verl |
| `verl/workers/critic/dp_critic` | `verl/workers/critic/dp_critic.py` | 10 | logging, os, torch, torch.distributed.fsdp, verl |
| `verl/workers/critic/megatron_critic` | `verl/workers/critic/megatron_critic.py` | 9 | functools, itertools, logging, omegaconf, os |
| `verl/workers/engine/__init__` | `verl/workers/engine/__init__.py` | 1 | .automodel, .fsdp, .megatron, .mindspeed, .torchtitan |
| `verl/workers/engine/automodel/__init__` | `verl/workers/engine/automodel/__init__.py` | 1 | - |
| `verl/workers/engine/automodel/transformer_impl` | `verl/workers/engine/automodel/transformer_impl.py` | 10 | contextlib, gc, logging, nemo_automodel.components.checkpoint.checkpointing, nemo_automodel.components.moe.megatron.moe_utils |
| `verl/workers/engine/automodel/utils` | `verl/workers/engine/automodel/utils.py` | 5 | megatron_fsdp.fully_shard, nemo_automodel._transformers.auto_model, nemo_automodel.components.distributed.mesh_utils, nemo_automodel.components.quantization.fp8, nemo_automodel.components.utils.compile_utils |
| `verl/workers/engine/base` | `verl/workers/engine/base.py` | 2 | abc, contextlib, tensordict, torch, typing |
| `verl/workers/engine/fsdp/__init__` | `verl/workers/engine/fsdp/__init__.py` | 1 | - |
| `verl/workers/engine/fsdp/transformer_impl` | `verl/workers/engine/fsdp/transformer_impl.py` | 19 | contextlib, gc, glob, logging, os |
| `verl/workers/engine/fsdp/utils` | `verl/workers/engine/fsdp/utils.py` | 2 | logging, os, torch, torch.distributed.device_mesh, torch.distributed.fsdp |
| `verl/workers/engine/megatron/__init__` | `verl/workers/engine/megatron/__init__.py` | 2 | os |
| `verl/workers/engine/megatron/transformer_impl` | `verl/workers/engine/megatron/transformer_impl.py` | 24 | functools, logging, megatron.core.transformer.enums, omegaconf, os |
| `verl/workers/engine/megatron/utils` | `verl/workers/engine/megatron/utils.py` | 2 | numpy, random, torch |
| `verl/workers/engine/mindspeed/__init__` | `verl/workers/engine/mindspeed/__init__.py` | 1 | - |
| `verl/workers/engine/mindspeed/transformer_impl` | `verl/workers/engine/mindspeed/transformer_impl.py` | 2 | ..megatron, logging, mindspeed.megatron_adaptor, os |
| `verl/workers/engine/torchtitan/__init__` | `verl/workers/engine/torchtitan/__init__.py` | 1 | - |
| `verl/workers/engine/torchtitan/transformer_impl` | `verl/workers/engine/torchtitan/transformer_impl.py` | 12 | contextlib, gc, importlib, logging, os |
| `verl/workers/engine/torchtitan/utils` | `verl/workers/engine/torchtitan/utils.py` | 2 | collections, collections.abc, dataclasses, importlib, logging |
| `verl/workers/engine/utils` | `verl/workers/engine/utils.py` | 5 | numpy, os, random, tensordict, torch |
| `verl/workers/engine/veomni/__init__` | `verl/workers/engine/veomni/__init__.py` | 1 | - |
| `verl/workers/engine/veomni/transformer_impl` | `verl/workers/engine/veomni/transformer_impl.py` | 11 | dataclasses, logging, tensordict, torch, torch.distributed.tensor |
| `verl/workers/engine/veomni/utils` | `verl/workers/engine/veomni/utils.py` | 1 | torch, torch.distributed.fsdp._fully_shard._fsdp_common, torch.distributed.fsdp._fully_shard._fsdp_state |
| `verl/workers/engine_workers` | `verl/workers/engine_workers.py` | 17 | codetiming, contextlib, copy, functools, itertools |
| `verl/workers/fsdp_workers` | `verl/workers/fsdp_workers.py` | 29 | codetiming, contextlib, dataclasses, datetime, glob |
| `verl/workers/megatron_workers` | `verl/workers/megatron_workers.py` | 30 | codetiming, contextlib, datetime, importlib, logging |
| `verl/workers/reward_manager/__init__` | `verl/workers/reward_manager/__init__.py` | 6 | - |
| `verl/workers/reward_manager/abstract` | `verl/workers/reward_manager/abstract.py` | 1 | abc, torch, typing |
| `verl/workers/reward_manager/batch` | `verl/workers/reward_manager/batch.py` | 1 | collections, torch, typing, verl, verl.workers.reward_manager |
| `verl/workers/reward_manager/dapo` | `verl/workers/reward_manager/dapo.py` | 1 | collections, torch, verl, verl.utils.reward_score, verl.workers.reward_manager |
| `verl/workers/reward_manager/naive` | `verl/workers/reward_manager/naive.py` | 1 | collections, torch, typing, verl, verl.utils.reward_score |
| `verl/workers/reward_manager/prime` | `verl/workers/reward_manager/prime.py` | 2 | asyncio, concurrent.futures, functools, psutil, torch |
| `verl/workers/reward_manager/registry` | `verl/workers/reward_manager/registry.py` | 1 | typing |
| `verl/workers/rollout/__init__` | `verl/workers/rollout/__init__.py` | 4 | - |
| `verl/workers/rollout/base` | `verl/workers/rollout/base.py` | 2 | abc, importlib, torch, torch.distributed.device_mesh, typing |
| `verl/workers/rollout/hf_rollout` | `verl/workers/rollout/hf_rollout.py` | 4 | contextlib, tensordict, torch, torch.distributed.fsdp, transformers |
| `verl/workers/rollout/naive/__init__` | `verl/workers/rollout/naive/__init__.py` | 1 | - |
| `verl/workers/rollout/naive/naive_rollout` | `verl/workers/rollout/naive/naive_rollout.py` | 2 | tensordict, torch, torch.nn.functional, verl |
| `verl/workers/rollout/replica` | `verl/workers/rollout/replica.py` | 9 | abc, asyncio, enum, logging, omegaconf |
| `verl/workers/rollout/schemas` | `verl/workers/rollout/schemas.py` | 3 | difflib, enum, logging, os, pydantic |
| `verl/workers/rollout/sglang_rollout/__init__` | `verl/workers/rollout/sglang_rollout/__init__.py` | 0 | - |
| `verl/workers/rollout/sglang_rollout/async_sglang_server` | `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | 10 | asyncio, dataclasses, json, logging, os |
| `verl/workers/rollout/sglang_rollout/http_server_engine` | `verl/workers/rollout/sglang_rollout/http_server_engine.py` | 1 | aiohttp, asyncio, base64, contextlib, logging |
| `verl/workers/rollout/sglang_rollout/sglang_rollout` | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | 8 | __future__, dataclasses, logging, multiprocessing, os |
| `verl/workers/rollout/sglang_rollout/utils` | `verl/workers/rollout/sglang_rollout/utils.py` | 3 | numpy, pickle, torch, typing |
| `verl/workers/rollout/tokenizer` | `verl/workers/rollout/tokenizer.py` | 0 | abc, numpy, torch |
| `verl/workers/rollout/trtllm_rollout/trtllm_async_server` | `verl/workers/rollout/trtllm_rollout/trtllm_async_server.py` | 8 | asyncio, inspect, logging, omegaconf, os |
| `verl/workers/rollout/trtllm_rollout/trtllm_rollout` | `verl/workers/rollout/trtllm_rollout/trtllm_rollout.py` | 7 | __future__, aiohttp, asyncio, base64, contextlib |
| `verl/workers/rollout/trtllm_rollout/trtllm_worker_extension` | `verl/workers/rollout/trtllm_rollout/trtllm_worker_extension.py` | 1 | base64, inspect, tensorrt_llm, tensorrt_llm._ray_utils, tensorrt_llm._torch.modules.fused_moe.moe_load_balancer |
| `verl/workers/rollout/utils` | `verl/workers/rollout/utils.py` | 0 | asyncio, fastapi, logging, numpy, uvicorn |
| `verl/workers/rollout/vllm_rollout/__init__` | `verl/workers/rollout/vllm_rollout/__init__.py` | 1 | importlib.metadata, os, re |
| `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer` | `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | 2 | gc, logging, multiprocessing, os, torch |
| `verl/workers/rollout/vllm_rollout/utils` | `verl/workers/rollout/vllm_rollout/utils.py` | 6 | ctypes, json, logging, os, platform |
| `verl/workers/rollout/vllm_rollout/vllm_async_server` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 13 | argparse, asyncio, inspect, json, logging |
| `verl/workers/rollout/vllm_rollout/vllm_omni_async_server` | `verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py` | 7 | argparse, dataclasses, logging, ray, torchvision.transforms |
| `verl/workers/rollout/vllm_rollout/vllm_rollout` | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 5 | logging, os, packaging, ray, time |
| `verl/workers/sharding_manager/__init__` | `verl/workers/sharding_manager/__init__.py` | 0 | - |
| `verl/workers/sharding_manager/base` | `verl/workers/sharding_manager/base.py` | 0 | verl |
| `verl/workers/sharding_manager/fsdp_ulysses` | `verl/workers/sharding_manager/fsdp_ulysses.py` | 3 | torch.distributed.device_mesh, verl |
| `verl/workers/utils/__init__` | `verl/workers/utils/__init__.py` | 0 | - |
| `verl/workers/utils/losses` | `verl/workers/utils/losses.py` | 6 | tensordict, torch, verl.utils.metric |
| `verl/workers/utils/padding` | `verl/workers/utils/padding.py` | 2 | tensordict, torch, torch.nn.functional |
