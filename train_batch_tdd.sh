get_free_gpu() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print NR-1 " " $1}' | sort -k2 -n -r | awk '{print $1}' | head -n 1
}

session=ETD

tmux has-session -t $session 2>/dev/null || tmux new-session -d -s $session

envs=("ObstructedMaze-1Q")
seeds=(0 1 2)
mtds=("TDD" "NovelD" "RND" "NoModel")
int_rew_coef=1e-2
loss_fns=("infonce_symmetric")
energy_fns=("mrn_pot")

for env in "${envs[@]}"
do
    for energy_fn in "${energy_fns[@]}"
    do
        for mtd in "${mtds[@]}"
        do
            for seed in "${seeds[@]}"
                do
                    exp_name="${mtd}-int${int_rew_coef}-ent_coef1e-3"
                    tmux select-window -t "$session":"$exp_name-$env-$seed" || tmux new-window -n "$exp_name-$env-$seed" -t $session
                    # Get the ID of the most free GPU
                    free_gpu=$(get_free_gpu)

                    tmux send-keys -t "$session":"$exp_name-$env-$seed" "conda activate tdd" C-m
                    tmux send-keys -t "$session":"$exp_name-$env-$seed" "CUDA_VISIBLE_DEVICES=$free_gpu PYTHONPATH='./' python3 src/train.py \
                    --game_name=$env \
                    --run_id=$seed \
                    --int_rew_source=${mtd} \
                    --env_source=minigrid \
                    --exp_name=$exp_name \
                    --use_wandb=1 \
                    --int_rew_coef=${int_rew_coef} \
                    --ext_rew_coef=10.0 \
                    --model_learning_rate=3e-4 \
                    --ent_coef=1e-2 \
                    --max_grad_norm=0.5 \
                    --tdd_aggregate_fn=min \
                    --tdd_energy_fn=$energy_fn \
                    --tdd_loss_fn=infonce_symmetric \
                    --tdd_logsumexp_coef=0 \
                    --offpolicy_data=0 \
                    --model_n_epochs=4 \
                    --n_steps=256 \
                    --num_processes=16 \
                    --use_model_rnn=0 \
                    --record_video=0 \
                    --enable_plotting=0 \
                    --model_mlp_layers=1 \
                    --log_explored_states=1 \
                    --policy_cnn_type=1 \
                    --model_cnn_type=1 \
                    --total_steps=10_000_000 \
                    --discount=0.99 \
                    --features_dim=64  \
                    --model_features_dim=64  \
                    --model_cnn_norm=LayerNorm \
                    --model_mlp_norm=LayerNorm \
                    --policy_cnn_norm=LayerNorm \
                    --policy_mlp_norm=LayerNorm \
                    " C-m
                    # tmux send-keys -t "$session":"$exp_name-$env" "echo success" C-m
                sleep 1
                done
            sleep 30
        done
    done
done

# # 进入 tmux 会话（可选）
# tmux attach-session -d
