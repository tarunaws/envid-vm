#!/bin/bash
#
# Translation Job Monitor
# Monitors a specific job's translation progress in real-time
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id>"
    echo "Example: $0 abhayPromo"
    exit 1
fi

BACKEND_URL="${BACKEND_URL:-http://localhost/backend}"
TRANSLATE_URL="${TRANSLATE_URL:-http://localhost:5102}"

echo "=================================================="
echo "  Translation Job Monitor - Job ID: $JOB_ID"
echo "=================================================="
echo

# Function to get job status
get_job_status() {
    curl -s "${BACKEND_URL}/jobs/${JOB_ID}" 2>/dev/null || echo "{}"
}

# Function to get translation health
get_translation_health() {
    curl -s "${TRANSLATE_URL}/health" 2>/dev/null || echo "{}"
}

# Function to get docker stats
get_docker_stats() {
    docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" translate-indian translate-international backend 2>/dev/null || echo ""
}

# Function to display job progress
display_job_progress() {
    local job_data=$1
    
    if command -v jq >/dev/null 2>&1; then
        local status=$(echo "$job_data" | jq -r '.status // "unknown"')
        local progress=$(echo "$job_data" | jq -r '.progress // 0')
        local message=$(echo "$job_data" | jq -r '.message // ""')
        local step=$(echo "$job_data" | jq -r '.current_step // ""')
        
        echo -e "${CYAN}Status:${NC} $status | ${CYAN}Progress:${NC} ${progress}% | ${CYAN}Step:${NC} $step"
        if [ -n "$message" ]; then
            echo -e "${CYAN}Message:${NC} $message"
        fi
        
        # Show translation step details
        local translate_step=$(echo "$job_data" | jq -r '.steps.translate_output // empty')
        if [ -n "$translate_step" ]; then
            local trans_status=$(echo "$translate_step" | jq -r '.status // ""')
            local trans_percent=$(echo "$translate_step" | jq -r '.percent // 0')
            local trans_message=$(echo "$translate_step" | jq -r '.message // ""')
            
            if [ "$trans_status" = "running" ] || [ "$trans_status" = "completed" ]; then
                echo -e "${BLUE}Translation:${NC} $trans_status ($trans_percent%) - $trans_message"
            fi
        fi
    else
        echo "$job_data"
    fi
}

# Function to display translation health
display_translation_health() {
    local health_data=$1
    
    if command -v jq >/dev/null 2>&1; then
        local ok=$(echo "$health_data" | jq -r '.ok // false')
        local mem_usage=$(echo "$health_data" | jq -r '.memory_usage_pct // 0')
        local mem_available=$(echo "$health_data" | jq -r '.memory_available_gb // 0')
        local models_loaded=$(echo "$health_data" | jq -r '.models_loaded // 0')
        local max_concurrent=$(echo "$health_data" | jq -r '.max_concurrent_requests // 0')
        
        local status_color=$GREEN
        if [ "$ok" != "true" ]; then
            status_color=$RED
        elif (( $(echo "$mem_usage > 85" | bc -l) )); then
            status_color=$YELLOW
        fi
        
        echo -e "${status_color}Translation Service:${NC} OK=$ok | Memory: ${mem_usage}% | Available: ${mem_available}GB | Models: ${models_loaded} | Queue: ${max_concurrent}"
    else
        echo "$health_data"
    fi
}

# Main monitoring loop
iteration=0
while true; do
    clear
    echo "=================================================="
    echo -e "  ${CYAN}Translation Job Monitor${NC} - Job ID: ${YELLOW}$JOB_ID${NC}"
    echo -e "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    echo
    
    # Get job status
    echo -e "${BLUE}► Job Status:${NC}"
    job_data=$(get_job_status)
    display_job_progress "$job_data"
    echo
    
    # Check if job is complete
    if command -v jq >/dev/null 2>&1; then
        status=$(echo "$job_data" | jq -r '.status // "unknown"')
        if [ "$status" = "completed" ] || [ "$status" = "failed" ]; then
            echo
            echo -e "${GREEN}Job finished with status: $status${NC}"
            
            # Show final translation results
            translations=$(echo "$job_data" | jq -r '.artifacts.translations // empty')
            if [ -n "$translations" ]; then
                echo
                echo -e "${CYAN}Translations produced:${NC}"
                echo "$translations" | jq -r 'keys[]' | while read lang; do
                    echo "  - $lang"
                done
            fi
            break
        fi
    fi
    
    # Get translation service health
    echo -e "${BLUE}► Translation Service Health:${NC}"
    health_data=$(get_translation_health)
    display_translation_health "$health_data"
    echo
    
    # Get Docker stats
    echo -e "${BLUE}► Container Resources:${NC}"
    echo "Container                CPU       Memory Usage    Mem %"
    echo "-------------------------------------------------------"
    get_docker_stats
    echo
    
    # Get recent logs
    echo -e "${BLUE}► Recent Translation Logs (last 5 lines):${NC}"
    docker logs translate-indian --tail 5 2>&1 | grep -v "^$" || echo "  No recent logs"
    echo
    
    # Show backend translation errors if any
    echo -e "${BLUE}► Backend Translation Status:${NC}"
    docker logs backend --tail 20 2>&1 | grep -i "translat\|retry\|503" | tail -5 || echo "  No translation activity"
    echo
    
    echo -e "${CYAN}Monitoring... (refresh every 5s, Ctrl+C to stop)${NC}"
    echo -e "${YELLOW}Iteration: $((++iteration))${NC}"
    
    sleep 5
done

echo
echo "=================================================="
echo "  Monitoring Complete"
echo "=================================================="
