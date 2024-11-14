#!/bin/bash

k_folds=$(python -c "from config import k_folds; print(k_folds)")
model_name=$(python -c "from config import model_name; print(model_name)")
dataset_name=$(python -c "from config import dataset_name; print(dataset_name)")
# strategy=$(python -c "from config import strategy; print(strategy)")
drifting_type=$(python -c "from config import drifting_type; print(drifting_type)")
non_iid_type=$(python -c "from config import non_iid_type; print(non_iid_type)")
n_clients=$(python -c "from config import n_clients; print(n_clients)")
n_rounds=$(python -c "from config import n_rounds; print(n_rounds)")

# Function to format seconds into HH:MM:SS
format_time() {
    local T=$1
    local H=$((T/3600))
    local M=$(( (T%3600)/60 ))
    local S=$((T%60))
    printf "%02d:%02d:%02d" $H $M $S
}

# Initialize counters for total folds
total_scalings=4
total_strategies=4






# fedrc
strategy="IFCA"

# Capture the overall start time
overall_start_time=$(date +%s)

# Initialize counters for total folds
total_folds=$((total_scalings * k_folds))
completed_folds=0
total_elapsed=0

for scaling in $(seq 1 4); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"


    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Capture the start time for the fold
        fold_start_time=$(date +%s)

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        python generate_datasets.py --fold "$fold" --scaling "$scaling"

        python start_experiment.py --fold "$fold" --method "$strategy"

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        sleep 3
    
        # Capture the end time for the fold
        fold_end_time=$(date +%s)
        fold_duration=$((fold_end_time - fold_start_time))
        total_elapsed=$((total_elapsed + fold_duration))
        completed_folds=$((completed_folds + 1))

        # Calculate average time per fold
        avg_time=$((total_elapsed / completed_folds))

        # Calculate remaining folds
        remaining_folds=$((total_folds - completed_folds))

        # Estimate remaining time for the current strategy
        estimated_remaining=$((avg_time * remaining_folds))

        # Format times
        formatted_fold_time=$(format_time $fold_duration)
        formatted_estimated_remaining=$(format_time $estimated_remaining)

        echo -e "\n\033[1;32mFold $completed_folds completed in $formatted_fold_time.\033[0m"
        echo -e "\033[1;33mEstimated time remaining (for current strategy): $formatted_estimated_remaining.\033[0m"

        # Estimate total remaining time for all strategies
        total_remaining_for_all_strategies=$((estimated_remaining + (avg_time * k_folds * total_scalings * (total_strategies - 1))))
        formatted_total_remaining_for_all_strategies=$(format_time $total_remaining_for_all_strategies)
        echo -e "\033[1;35mEstimated total time remaining for all strategies: $formatted_total_remaining_for_all_strategies.\033[0m\n"

    done

    # K-Fold evaluation, if k_folds > 1
    if [ "$k_folds" -gt 1 ]; then

        echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
        # Averaging the results of all folds
        python average_results.py --scaling "$scaling" --strategy "$strategy"

    fi

done

echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"





# fedrc
strategy="fedrc"

# Capture the overall start time
overall_start_time=$(date +%s)

# Initialize counters for total folds
total_strategies=$((total_strategies - 1))
total_folds=$((total_scalings * k_folds))
completed_folds=0
total_elapsed=0

for scaling in $(seq 1 4); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"


    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Capture the start time for the fold
        fold_start_time=$(date +%s)

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        python generate_datasets.py --fold "$fold" --scaling "$scaling"

        python start_experiment.py --fold "$fold" --method "$strategy"

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        sleep 3

        # Capture the end time for the fold
        fold_end_time=$(date +%s)
        fold_duration=$((fold_end_time - fold_start_time))
        total_elapsed=$((total_elapsed + fold_duration))
        completed_folds=$((completed_folds + 1))

        # Calculate average time per fold
        avg_time=$((total_elapsed / completed_folds))

        # Calculate remaining folds
        remaining_folds=$((total_folds - completed_folds))

        # Estimate remaining time for the current strategy
        estimated_remaining=$((avg_time * remaining_folds))

        # Format times
        formatted_fold_time=$(format_time $fold_duration)
        formatted_estimated_remaining=$(format_time $estimated_remaining)

        echo -e "\n\033[1;32mFold $completed_folds completed in $formatted_fold_time.\033[0m"
        echo -e "\033[1;33mEstimated time remaining (for current strategy): $formatted_estimated_remaining.\033[0m"

        # Estimate total remaining time for all strategies
        total_remaining_for_all_strategies=$((estimated_remaining + (avg_time * k_folds * total_scalings * (total_strategies - 1))))
        formatted_total_remaining_for_all_strategies=$(format_time $total_remaining_for_all_strategies)
        echo -e "\033[1;35mEstimated total time remaining for all strategies: $formatted_total_remaining_for_all_strategies.\033[0m\n"


    done

    # K-Fold evaluation, if k_folds > 1
    if [ "$k_folds" -gt 1 ]; then

        echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
        # Averaging the results of all folds
        python average_results.py --scaling "$scaling" --strategy "$strategy"

    fi

done

echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"






# FedEM
strategy="FedEM"

# Capture the overall start time
overall_start_time=$(date +%s)

# Initialize counters for total folds
total_strategies=$((total_strategies - 1))
total_folds=$((total_scalings * k_folds))
completed_folds=0
total_elapsed=0

for scaling in $(seq 1 4); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"


    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Capture the start time for the fold
        fold_start_time=$(date +%s)

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        python generate_datasets.py --fold "$fold" --scaling "$scaling"

        python start_experiment.py --fold "$fold" --method "$strategy"

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        sleep 3

        # Capture the end time for the fold
        fold_end_time=$(date +%s)
        fold_duration=$((fold_end_time - fold_start_time))
        total_elapsed=$((total_elapsed + fold_duration))
        completed_folds=$((completed_folds + 1))

        # Calculate average time per fold
        avg_time=$((total_elapsed / completed_folds))

        # Calculate remaining folds
        remaining_folds=$((total_folds - completed_folds))

        # Estimate remaining time for the current strategy
        estimated_remaining=$((avg_time * remaining_folds))

        # Format times
        formatted_fold_time=$(format_time $fold_duration)
        formatted_estimated_remaining=$(format_time $estimated_remaining)

        echo -e "\n\033[1;32mFold $completed_folds completed in $formatted_fold_time.\033[0m"
        echo -e "\033[1;33mEstimated time remaining (for current strategy): $formatted_estimated_remaining.\033[0m"

        # Estimate total remaining time for all strategies
        total_remaining_for_all_strategies=$((estimated_remaining + (avg_time * k_folds * total_scalings * (total_strategies - 1))))
        formatted_total_remaining_for_all_strategies=$(format_time $total_remaining_for_all_strategies)
        echo -e "\033[1;35mEstimated total time remaining for all strategies: $formatted_total_remaining_for_all_strategies.\033[0m\n"


    done

    # K-Fold evaluation, if k_folds > 1
    if [ "$k_folds" -gt 1 ]; then

        echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
        # Averaging the results of all folds
        python average_results.py --scaling "$scaling" --strategy "$strategy"

    fi

done

echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"







# FeSEM
strategy="FeSEM"

# Capture the overall start time
overall_start_time=$(date +%s)

# Initialize counters for total folds
total_strategies=$((total_strategies - 1))
total_folds=$((total_scalings * k_folds))
completed_folds=0
total_elapsed=0

for scaling in $(seq 1 4); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"


    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Capture the start time for the fold
        fold_start_time=$(date +%s)

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        python generate_datasets.py --fold "$fold" --scaling "$scaling"

        python start_experiment.py --fold "$fold" --method "$strategy"

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        sleep 3

        # Capture the end time for the fold
        fold_end_time=$(date +%s)
        fold_duration=$((fold_end_time - fold_start_time))
        total_elapsed=$((total_elapsed + fold_duration))
        completed_folds=$((completed_folds + 1))

        # Calculate average time per fold
        avg_time=$((total_elapsed / completed_folds))

        # Calculate remaining folds
        remaining_folds=$((total_folds - completed_folds))

        # Estimate remaining time for the current strategy
        estimated_remaining=$((avg_time * remaining_folds))

        # Format times
        formatted_fold_time=$(format_time $fold_duration)
        formatted_estimated_remaining=$(format_time $estimated_remaining)

        echo -e "\n\033[1;32mFold $completed_folds completed in $formatted_fold_time.\033[0m"
        echo -e "\033[1;33mEstimated time remaining (for current strategy): $formatted_estimated_remaining.\033[0m"

        # Estimate total remaining time for all strategies
        total_remaining_for_all_strategies=$((estimated_remaining + (avg_time * k_folds * total_scalings * (total_strategies - 1))))
        formatted_total_remaining_for_all_strategies=$(format_time $total_remaining_for_all_strategies)
        echo -e "\033[1;35mEstimated total time remaining for all strategies: $formatted_total_remaining_for_all_strategies.\033[0m\n"


    done

    # K-Fold evaluation, if k_folds > 1
    if [ "$k_folds" -gt 1 ]; then

        echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
        # Averaging the results of all folds
        python average_results.py --scaling "$scaling" --strategy "$strategy"

    fi

done

echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"
# kill
trap - SIGTERM && kill -- -$$






