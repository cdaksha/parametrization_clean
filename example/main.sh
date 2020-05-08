#!/usr/bin/env bash
#
# Example usage of automating the command line interface to run a job that continues many generations.
# The role of the Python application is to propagate from one generation to the next.
# The role of this wrapper Bash script is to submit the ReaxFF optimizations that were created, monitor them, and
# call the Python application once more after the optimizations are completed.
# This example uses a SLURM-style ReaxFF optimization submission script.

# SETUP
# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__root="$(cd "$(dirname "${__dir}")" && pwd)"


#######################################
# CONSTANTS
#######################################
# COMMAND LINE INTERFACE CONSTANTS
#
# GENERATION_NUM: int
#   Number of the generation of the genetic algorithm at which to start the job.
#   As generations go on, the generation number is dynamically changed to reflect progression of the generational GA.
#   If GENERATION_NUM > 1, the genetic population from (GENERATION_NUM - 1) is assumed to exist and
#   is read to retrieve the previous generation's population.
#
# MAX_GENERATION_NUM: int
#   Generation number at which to stop the generational genetic algorithm. For example, if MAX_GENERATION_NUM = 501,
#   then the first 500 generations of the generational GA will run, after which the script will stop.
#
# TRAINING_PATH: string/file path
#   Location of reference training set directory with  model & other data files relevant to current parameter
#   optimization.
#   Must contain at least the ffield, geo, params, control, and trainset.in files.
#
# POPULATION_PATH: string/file path
#   Location of path at which to output files for generational genetic algorithm. Different generations are created in
#   directories corresponding to their generation number (ex. generation-1/, generation-2/, ...). Each generation folder
#   contains "children" cases corresponding to that generation: child-1/, child-2/, ..., etc. Each "child" is a ReaxFF
#   optimization run that must be submitted and run.
#
# CONFIG_PATH: string/file path, optional, otherwise set to `""`
#	Location of user configuration JSON file to specify algorithm parameters desired for usage.
#
# POPULATION_SIZE: int
#   Number of individuals one wishes to create for each generation in the generational genetic algorithm.
#   For example, if POPULATION_SIZE = 50, then 50 children will be created in each generation.
#######################################
GENERATION_NUM=1
MAX_GENERATION_NUM=501
TRAINING_PATH=""
POPULATION_PATH=""
CONFIG_PATH=""
POPULATION_SIZE=30


# REAXFF JOB AUTOMATION CONSTANTS
USER=daksha  # user ID to use for checking number of remaining jobs
JOB_NAME="RUN-1"  # job name to use for ReaxFF optimizations
MAX_JOB_TIME="0-00:30:00"  # maximum allowable ReaxFF optimization run time
JOB_INTERVAL=30  # How often the script should check for the job status in seconds.


#######################################
# FUNCTIONS
#######################################
# Wrapper function to run Python command line interface.
# Initialize (GENERATION_NUM=1) or propagate (GENERATION_NUM>1) a generation
# in the generational GA. Output generational summary & ANN statistics (if ANN is enabled).
#######################################
main() {
cli --generation_number ${GENERATION_NUM} --training_path "${TRAINING_PATH}" --population_path "${POPULATION_PATH}" \
    --config_path "${CONFIG_PATH}"
}

#######################################
# Create SLURM submission files & submit for each folder in generation-$GENERATION_NUM.
# *** DOES NOT USE SLURM JOB ARRAY ***
#######################################
submitReaxFFOptimizations() {
total_num_files=$POPULATION_SIZE

# Safety measure
if [[ ! -d "${POPULATION_PATH}/generation-${GENERATION_NUM}" ]]; then
  echo "generation-${GENERATION_NUM} does not exist at ${POPULATION_PATH}."
  echo "Exiting..."
  exit
fi
echo "Submitting simulations for generation-${GENERATION_NUM}..."

for ((i = 0; i < ${total_num_files}; i++)); do
INPUT="child-${i}"
cd "${POPULATION_PATH}/generation-${GENERATION_NUM}/$INPUT" || exit

cat > $INPUT.sh << EOF
#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
### SBATCH --cpus-per-task=1
#SBATCH --partition=ccm_gillespi
### SBATCH --partition=standard
#SBATCH --time=${MAX_JOB_TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=GAt${i}.out
#SBATCH --error=GAt${i}error.out
### SBATCH --mail-user='${USER}@udel.edu'
### SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

. /opt/shared/slurm/templates/libexec/common.sh
vpkg_require reaxff/2.0.1:intel
#REAXFF_NO_BACKUPS=YES
. /work/ccm_gillespi/sw/reaxff/init_workdir.sh

time reac
rc=\$?
exit \$rc

EOF
chmod u+x $INPUT.sh
sbatch $INPUT.sh
done
# Go back to original directory
cd "${__dir}" || exit
}


#######################################
# RUNNING SCRIPT
#######################################
# Timer
SECONDS=0

# Scheduled job
while [ $GENERATION_NUM -le $MAX_GENERATION_NUM ]; do
    main
    submitReaxFFOptimizations

    num_optimizations_remaining=${POPULATION_SIZE}
    while [ "$num_optimizations_remaining" -gt 0 ]; do
        echo "Number of remaining jobs: ${num_optimizations_remaining}..."
        sleep ${JOB_INTERVAL}
        num_optimizations_remaining=$(squeue -u ${USER} | grep -c ${JOB_NAME})
    done

    GENERATION_NUM=$((GENERATION_NUM + 1))
    echo "Time Elapsed: ${SECONDS} seconds."
done

# To gather data for the very last generation
main

echo "Job Completed!"
echo "Time Elapsed: ${SECONDS} seconds."
