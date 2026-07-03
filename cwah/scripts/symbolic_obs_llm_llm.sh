pid=$(lsof -t -i :6314)
if [ -n "$pid" ]; then
kill -9 $pid
fi
python testing_agents/test_symbolic_LLMs.py \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--mode LLMs_comm_gpt-5-mini \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6314 \
--lm_id gpt-5.4-mini \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--debug
