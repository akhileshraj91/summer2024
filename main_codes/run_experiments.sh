#!/bin/bash

# Experiment Runner Script for mild_RL_controller.py
# This script runs experiments with different applications and preference values

# Set script to exit on any error
set -e

# Configuration
SCRIPT_NAME="mild_RL_controller.py"
MODEL_PATH="trained_models/trained_network_weights_20250924_171320_all_preference_ones-stream-full_0.3_0.001.pth"
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Define applications to test
declare -a APPLICATIONS=(
    # "ones-stream-full"
    # "ones-stream-scale"
    # "ones-stream-triad"
    # "ones-stream-add"
    # "ones-stream-copy"
    # "ones-npb-ep"
    # "ones-npb-is"
    "ones-npb-ft"
    # "ones-npb-bt"
)

# Define preference ratios to test
declare -a PREFERENCES=(
    "[1,0]"
    "[0.95,0.05]"
    "[0.9,0.1]"
    "[0.85,0.15]"
    "[0.8,0.2]"
    "[0.75,0.25]"
    "[0.7,0.3]"
    "[0.65,0.35]"
    "[0.6,0.4]"
    "[0.55,0.45]"
    "[0.5,0.5]"
    "[0.45,0.55]"
    "[0.4,0.6]"
    "[0.35,0.65]"
    "[0.3,0.7]"
    "[0.25,0.75]"
    "[0.2,0.8]"
    "[0.15,0.85]"
    "[0.1,0.9]"
    "[0.05,0.95]"
    "[0,1]"
)

# Function to run a single experiment
run_experiment() {
    local app="$1"
    local preference="$2"
    local log_file="$LOG_DIR/experiment_${app}_${preference//[\[\],]/_}_${TIMESTAMP}.log"
    
    echo "========================================="
    echo "Running experiment:"
    echo "  Application: $app"
    echo "  Preference: $preference"
    echo "  Log file: $log_file"
    echo "========================================="
    
    # Run the experiment and capture output with timeout
    if python3 "$SCRIPT_NAME" -a "$app" -p "$MODEL_PATH" -r "$preference" > "$log_file" 2>&1; then
        echo "✅ Experiment completed successfully"
        echo "Results saved to: $log_file"
    else
        local exit_code=$?
        echo "❌ Experiment failed or timed out (exit code: $exit_code)"
        echo "Check log file for details: $log_file"
        
        # Clean up any stuck Python processes
        pkill -f "mild_RL_controller.py" 2>/dev/null || true
        
        return 1
    fi
    
    echo ""
}

# Function to run all combinations
run_all_experiments() {
    local total_experiments=$((${#APPLICATIONS[@]} * ${#PREFERENCES[@]}))
    local current_experiment=0
    
    echo "Starting experiment batch at $(date)"
    echo "Total experiments to run: $total_experiments"
    echo "Applications: ${#APPLICATIONS[@]}"
    echo "Preference ratios: ${#PREFERENCES[@]}"
    echo ""
    
    for app in "${APPLICATIONS[@]}"; do
        for preference in "${PREFERENCES[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "[$current_experiment/$total_experiments] Running experiment..."
            
            if ! run_experiment "$app" "$preference"; then
                echo "⚠️  Experiment failed, but continuing with remaining experiments..."
            fi
        done
    done
    
    echo "========================================="
    echo "All experiments completed at $(date)"
    echo "Check the $LOG_DIR directory for results"
    echo "========================================="
}

# Function to run specific application
run_application_experiments() {
    local target_app="$1"
    echo "Running experiments for application: $target_app"
    echo ""
    
    for preference in "${PREFERENCES[@]}"; do
        run_experiment "$target_app" "$preference"
    done
}

# Function to run specific preference
run_preference_experiments() {
    local target_preference="$1"
    echo "Running experiments for preference: $target_preference"
    echo ""
    
    for app in "${APPLICATIONS[@]}"; do
        run_experiment "$app" "$target_preference"
    done
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -a, --app APP           Run experiments for specific application only"
    echo "  -p, --preference PREF   Run experiments for specific preference only"
    echo "  -l, --list              List available applications and preferences"
    echo "  --all                   Run all experiments (default)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run all experiments"
    echo "  $0 -a ones-stream-full  # Run only for ones-stream-full application"
    echo "  $0 -p '[1,0]'           # Run only for [1,0] preference"
    echo "  $0 --all                # Run all experiments (explicit)"
}

# Function to list available options
list_options() {
    echo "Available Applications:"
    for app in "${APPLICATIONS[@]}"; do
        echo "  - $app"
    done
    echo ""
    echo "Available Preference Ratios:"
    for pref in "${PREFERENCES[@]}"; do
        echo "  - $pref"
    done
}

# Main script logic
main() {
    case "${1:-}" in
        -h|--help)
            show_usage
            ;;
        -l|--list)
            list_options
            ;;
        -a|--app)
            if [[ -z "${2:-}" ]]; then
                echo "Error: Application name required after -a/--app"
                exit 1
            fi
            run_application_experiments "$2"
            ;;
        -p|--preference)
            if [[ -z "${2:-}" ]]; then
                echo "Error: Preference ratio required after -p/--preference"
                exit 1
            fi
            run_preference_experiments "$2"
            ;;
        --all|"")
            run_all_experiments
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
