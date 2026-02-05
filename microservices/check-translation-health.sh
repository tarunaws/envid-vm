#!/bin/bash
#
# Translation Service Health Check Script
# Tests and monitors translation service health and resource usage
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "  Translation Service Health Check"
echo "=================================================="
echo

# Function to check if service is running
check_service_running() {
    local service_name=$1
    if docker ps --format '{{.Names}}' | grep -q "^${service_name}$"; then
        echo -e "${GREEN}✓${NC} ${service_name} is running"
        return 0
    else
        echo -e "${RED}✗${NC} ${service_name} is not running"
        return 1
    fi
}

# Function to check endpoint health
check_endpoint() {
    local name=$1
    local url=$2
    
    echo -n "Checking ${name}... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${url}" 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}FAILED${NC} (HTTP $response)"
        return 1
    fi
}

# Function to get and display health details
get_health_details() {
    local name=$1
    local url=$2
    
    echo "--- ${name} Health Details ---"
    health_data=$(curl -s "${url}" 2>/dev/null || echo "{}")
    
    if command -v jq >/dev/null 2>&1; then
        echo "$health_data" | jq '.'
    else
        echo "$health_data"
    fi
    echo
}

# Function to check Docker stats
check_docker_stats() {
    local service_name=$1
    
    echo "--- ${service_name} Resource Usage ---"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" "${service_name}" 2>/dev/null || echo "Unable to get stats"
    echo
}

# Function to check memory limits
check_memory_limits() {
    local service_name=$1
    
    echo "--- ${service_name} Memory Limits ---"
    mem_limit=$(docker inspect "${service_name}" --format='{{.HostConfig.Memory}}' 2>/dev/null)
    
    if [ "$mem_limit" = "0" ]; then
        echo -e "${YELLOW}⚠${NC} No memory limit set"
    else
        mem_limit_gb=$(echo "scale=2; $mem_limit / 1024 / 1024 / 1024" | bc)
        echo -e "${GREEN}✓${NC} Memory limit: ${mem_limit_gb} GB"
    fi
    echo
}

# Function to test translation
test_translation() {
    local url=$1
    local lang=$2
    
    echo -n "Testing translation to ${lang}... "
    
    payload='{"text":"Hello world","source_lang":"en","target_lang":"'${lang}'"}'
    response=$(curl -s -X POST "${url}/translate" \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        --max-time 30 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q "translated_text"; then
        echo -e "${GREEN}OK${NC}"
        if command -v jq >/dev/null 2>&1; then
            echo "  Result: $(echo "$response" | jq -r '.translated_text')"
        fi
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Response: $response"
        return 1
    fi
}

# Main checks
echo "1. Container Status"
echo "-------------------"
check_service_running "translate-indian"
check_service_running "translate-international"
echo

echo "2. Service Endpoints"
echo "--------------------"
check_endpoint "IndicTrans2" "http://localhost:5102/health"
check_endpoint "LibreTranslate" "http://localhost:5000/health"
check_endpoint "LanguageTool" "http://localhost:8010/v2/check"
echo

echo "3. Health Details"
echo "-----------------"
if check_service_running "translate-indian" >/dev/null 2>&1; then
    get_health_details "IndicTrans2" "http://localhost:5102/health"
fi

if check_service_running "translate-international" >/dev/null 2>&1; then
    get_health_details "LibreTranslate" "http://localhost:5000/health"
fi

echo "4. Resource Usage"
echo "-----------------"
if check_service_running "translate-indian" >/dev/null 2>&1; then
    check_docker_stats "translate-indian"
    check_memory_limits "translate-indian"
fi

if check_service_running "translate-international" >/dev/null 2>&1; then
    check_docker_stats "translate-international"
    check_memory_limits "translate-international"
fi

echo "5. Translation Tests"
echo "--------------------"
if check_endpoint "IndicTrans2" "http://localhost:5102/health" >/dev/null 2>&1; then
    test_translation "http://localhost:5102" "hi"
    echo
fi

echo "=================================================="
echo "  Health Check Complete"
echo "=================================================="
echo
echo "Tips:"
echo "  - Watch logs: docker logs -f translate-indian"
echo "  - Monitor live: watch -n 2 'docker stats --no-stream translate-indian translate-international'"
echo "  - Check errors: docker logs translate-indian 2>&1 | grep -i error"
echo
