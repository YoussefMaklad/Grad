#!/bin/bash

# Function to manage FuzAgent-API service
function manage_llm_service() {
    local action=$1
    local service_dir="ChatAFL-CL2"
    local pid_file="$service_dir/service.pid"

    case "$action" in
        start)
            echo "Starting FuzAgent-API service..."
            cd $service_dir
            python3 main.py &
            echo $! > $pid_file
            cd ..
            echo "Waiting for service to initialize..."
            sleep 5
            echo "Service started on http://127.0.0.1:8000"
            ;;
        stop)
            if [ -f "$pid_file" ]; then
                echo "Stopping FuzAgent-API service..."
                kill $(cat "$pid_file")
                rm "$pid_file"
                echo "Service stopped"
            fi
            ;;
        status)
            if [ -f "$pid_file" ] && ps -p $(cat "$pid_file") > /dev/null; then
                echo "FuzAgent-API service is running"
                curl -s http://127.0.0.1:8000/health
            else
                echo "LLM service is not running"
            fi
            ;;
    esac
}

# Start the LLM service
manage_llm_service start

PFBENCH="$PWD/benchmark"
cd $PFBENCH

PATH=$PATH:$PFBENCH/scripts/execution:$PFBENCH/scripts/analysis
NUM_CONTAINERS=$1
TIMEOUT=$(( ${2:-1440} * 60))
SKIPCOUNT="${SKIPCOUNT:-1}"
TEST_TIMEOUT="${TEST_TIMEOUT:-5000}"

export TARGET_LIST=$3
export FUZZER_LIST=$4

if [[ "x$NUM_CONTAINERS" == "x" ]] || [[ "x$TIMEOUT" == "x" ]] || [[ "x$TARGET_LIST" == "x" ]] || [[ "x$FUZZER_LIST" == "x" ]]
then
    echo "Usage: $0 NUM_CONTAINERS TIMEOUT TARGET FUZZER"
    exit 1
fi

PFBENCH=$PFBENCH PATH=$PATH NUM_CONTAINERS=$NUM_CONTAINERS TIMEOUT=$TIMEOUT SKIPCOUNT=$SKIPCOUNT TEST_TIMEOUT=$TEST_TIMEOUT scripts/execution/profuzzbench_exec_all.sh ${TARGET_LIST} ${FUZZER_LIST}