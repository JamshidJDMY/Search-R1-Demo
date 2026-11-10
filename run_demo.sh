#!/usr/bin/env bash
# Support invocation with sh by switching to Bash before parsing Bash syntax.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# Start the retriever, wait for readiness, and launch the web demo.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Parse the web server address and port.
web_host=127.0.0.1
web_port=7860
while (( $# )); do
    case "$1" in
        --host|--port)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 1
            fi
            if [[ "$1" == --host ]]; then web_host=$2; else web_port=$2; fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_demo.sh [--host ADDRESS] [--port PORT]"
            echo "Run on your allocated GPU node; defaults: 127.0.0.1:7860."
            exit 0
            ;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done
if [[ ! "$web_port" =~ ^[0-9]{1,5}$ ]] || (( 10#$web_port < 1 || 10#$web_port > 65535 )); then
    echo "--port must be between 1 and 65535." >&2
    exit 1
fi
web_port=$((10#$web_port))

for command in conda curl nohup; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

for file in demo.py web_demo/index.html retrieval_launch.sh corpus/e5_Flat.index corpus/wiki-18.jsonl; do
    if [[ ! -r "$file" ]]; then
        echo "Required file missing or unreadable: $PWD/$file" >&2
        exit 1
    fi
done

gpu_node=$(hostname -s)
echo "Running on: $gpu_node (use your allocated GPU node, not the login node)."

docs_url=http://127.0.0.1:8000/docs
startup_timeout=${RETRIEVER_STARTUP_TIMEOUT:-1800}
if [[ ! "$startup_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "RETRIEVER_STARTUP_TIMEOUT must be a positive number of seconds." >&2
    exit 1
fi

is_ready() {
    local status
    status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$docs_url") || return 1
    [[ "$status" == 200 ]]
}

if is_ready; then
    echo "Retriever already reachable at $docs_url (HTTP 200)."
else
    nohup conda run --no-capture-output \
        bash retrieval_launch.sh > retriever.log 2>&1 < /dev/null &
    retriever_pid=$!
    echo "Started retriever (PID $retriever_pid). It will remain running after the demo stops."
    echo "View startup logs in another terminal: tail -f '$PWD/retriever.log'"

    deadline=$((SECONDS + startup_timeout))
    until is_ready; do
        if ! kill -0 "$retriever_pid" 2>/dev/null; then
            echo "Retriever exited during startup. Last log lines:" >&2
            tail -n 40 retriever.log >&2
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for the retriever after ${startup_timeout}s; see retriever.log." >&2
            echo "The retriever process has been left running (PID $retriever_pid)." >&2
            exit 1
        fi
        sleep 2
    done
    echo "Retriever reachable at $docs_url (HTTP 200)."
fi

echo "Starting the web demo in the active conda environment..."
conda run --no-capture-output python -u demo.py --host "$web_host" --port "$web_port"
