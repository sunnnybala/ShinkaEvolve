from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Configure the job execution environment
job_config = LocalJobConfig(
    eval_program_path="examples/grpo_countdown/evaluate.py",
    #conda_env="my_special_env",  # Optional: run in specific conda environment
)

# Configure the evolution database
db_config = DatabaseConfig(
    archive_size=20,
    num_archive_inspirations=4,
    num_islands=2,
    migration_interval=10,
)

# Configure the evolution parameters
evo_config = EvolutionConfig(
    num_generations=10,
    max_parallel_jobs=1,
    llm_models=["azure-gpt-5"],
    init_program_path="examples/grpo_countdown/initial.py",
    language="python",
    task_sys_msg=(
        "You are optimizing a GRPO Countdown evaluation program. Modify only the EVOLVE-BLOCK to"
        " improve accuracy at max_new_tokens=512. Preserve evaluation integrity and correctness."
    ),
)

# Run the evolution
runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)
runner.run()