#!/bin/bash
#
# Wrapper Bash Script that calls Python driver module, `main.py`, to run Genetic Algorithm (+ Artificial Neural Network)


#######################################
# GLOBAL CONSTANTS
#######################################
# GENERATION_NUM: int
#   Number of the current generation of the genetic algorithm. Used when creating parent file directory
#######################################
GENERATION_NUM=1  # Current generation number
MAX_GENERATIONS=501  # Final generation number 
USER=daksha  # user ID to use for checking number of remaining jobs & job cancellation - job cancellation is currently commented out 
JOB_NAME="RUN-1"  # job name to use for ReaxFF optimizations
MAX_JOB_TIME="0-00:10:00"  # maximum ReaxFF optimization run time

JOB_INTERVAL=30  # How often the job should be checked. Ex: every 30 seconds
JOB_THRESHOLD=0  # TODO - Currently, JOB_THRESHOLD > 0 has no compatibility with slurm job since slurm job ends up being cancelled
USE_SLURM_ARRAY=true  # if false, do not use slurm array to submit ReaxFF optimizations

#######################################
# Setting alias for convenience, since 'python' instead of 'python3' must be used for Caviness
# COMMENT OUT FOR PERSONAL COMPUTER
python3() {
  python "$@"
}
export -f python3
#######################################

# For saving ANN model
export HDF5_USE_FILE_LOCKING='FALSE'

# Retrieving constants from 'settings.py' config file
scripts_dir=$(pwd)
settings_path=${scripts_dir}/core/settings/config.yaml

POPULATION_PATH=$(grep 'population:' $settings_path | tail -n1 | awk '{ print $2}')
POP_SIZE=$(grep 'populationSize:' $settings_path | tail -n1 | awk '{ print $2}')


#######################################
# FUNCTIONS
#######################################
# Create SLURM submission files & submit for each folder in generation-$GENERATION_NUM.
# *** USES SLURM JOB ARRAY ***
#######################################
submitRunsUsingArray() {
total_num_files=$POP_SIZE

if [[ "${GENERATION_NUM}" == 1 ]]; then
  echo "First generation -> increasing job time to 30 minutes!"
  MAX_JOB_TIME="0-00:30:00"
fi

# Safety measure
if [[ ! -d "${POPULATION_PATH}/generation-${GENERATION_NUM}" ]]; then
  echo "generation-${GENERATION_NUM} does not exist at ${POPULATION_PATH}."
  echo "Exiting..."
  exit
fi
echo "Submitting simulations for generation-${GENERATION_NUM}..."

# Incorporating SLURM arrays for better organization
cd "${POPULATION_PATH}/generation-${GENERATION_NUM}/"

# For output/error logging
JOB_OUTPUT_DIR="00-JOB-LOGS"
mkdir ${JOB_OUTPUT_DIR}

cat > submit_array.sh << EOF
#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-$(($total_num_files - 1))
### SBATCH --cpus-per-task=1
#SBATCH --partition=ccm_gillespi
### SBATCH --partition=standard
#SBATCH --time=${MAX_JOB_TIME}
### SBATCH --job-name=GAt$i
#SBATCH --job-name=${JOB_NAME}
### SBATCH --output=GAt${i}.out
#SBATCH --output=./${JOB_OUTPUT_DIR}/slurm-%a.out
### SBATCH --error=GAt${i}error.out
#SBATCH --error=./${JOB_OUTPUT_DIR}/slurm-error-%a.out
### SBATCH --mail-user='${USER}@udel.edu'
### SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

. /opt/shared/slurm/templates/libexec/common.sh
vpkg_require reaxff/2.0.1:intel
#REAXFF_NO_BACKUPS=YES

cd \${SLURM_SUBMIT_DIR}/child-\${SLURM_ARRAY_TASK_ID}
. /work/ccm_gillespi/sw/reaxff/init_workdir.sh

time reac
rc=\$?
exit \$rc

EOF
chmod u+x submit_array.sh
sbatch submit_array.sh
# Go back to original directory
cd $scripts_dir
}


#######################################
# Create SLURM submission files & submit for each folder in generation-$GENERATION_NUM.
# *** DOES NOT USE SLURM JOB ARRAY ***
#######################################
submitRunsWithoutArray() {
total_num_files=$POP_SIZE

# Safety measure
if [[ ! -d "${POPULATION_PATH}/generation-${GENERATION_NUM}" ]]; then
  echo "generation-${GENERATION_NUM} does not exist at ${POPULATION_PATH}."
  echo "Exiting..."
  exit
fi
echo "Submitting simulations for generation-${GENERATION_NUM}..."

for ((i = 0; i < $total_num_files; i++)); do
INPUT="child-${i}"
cd "${POPULATION_PATH}/generation-${GENERATION_NUM}/$INPUT"

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
#sleep 1 # pause to be kind to the scheduler
done
# Go back to original directory
cd $scripts_dir
}


#######################################
# Create (FIRST_GENERATION = True) or Modify (FIRST_GENERATION = False)
# ANN, output the Master GA best case & Nested GA predicted best case,
# and output the predicted top 10% of best cases from Nested GA for next
# generation of Master GA.
# ANN is stored/modified at driver.NETWORK_FILE_PATH
# The predicted population from Nested GA is outputted in generation-(GENERATION_NUM + 1)
# Ex: If GENERATION_NUM = 1, then the predicted population is outputted to
# the folder with name 'generation-2'.
#
# To be used after a given generation's ReaxFF runs finish.
#######################################
main() {
python3 main.py ${GENERATION_NUM}
}


#######################################
# Create and submit jobs for Pahari's first stage of parameter reduction technique
#######################################
create_and_submit_param_reduction() {
output_dir="00-param-reduction"
python3 -c "import param_reduction; param_reduction.run_param_reduction('${output_dir}')"

total_num_files=`python3 -c 'import param_reduction; print(param_reduction.get_number_of_params())'`

# Safety measure
if [[ ! -d "${POPULATION_PATH}/${output_dir}" ]]; then
  echo "${output_dir} does not exist at ${POPULATION_PATH}."
  echo "Exiting..."
  #exit
fi
echo "Submitting simulations for ${output_dir}..."

# Incorporating SLURM arrays for better organization
cd "${POPULATION_PATH}/${output_dir}/"

# For output/error logging
JOB_OUTPUT_DIR="00-JOB-LOGS"
mkdir ${JOB_OUTPUT_DIR}

#cat > $INPUT.sh << EOF
cat > submit_array.sh << EOF
#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-$(($total_num_files - 1))
### SBATCH --cpus-per-task=1
#SBATCH --partition=ccm_gillespi
### SBATCH --partition=standard
#SBATCH --time=${MAX_JOB_TIME}
### SBATCH --job-name=GAt$i
#SBATCH --job-name=${JOB_NAME}
### SBATCH --output=GAt${i}.out
#SBATCH --output=./${JOB_OUTPUT_DIR}/slurm-%a.out
### SBATCH --error=GAt${i}error.out
#SBATCH --error=./${JOB_OUTPUT_DIR}/slurm-error-%a.out
### SBATCH --mail-user='daksha@udel.edu'
### SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

. /opt/shared/slurm/templates/libexec/common.sh
vpkg_require reaxff/2.0.1:intel
#REAXFF_NO_BACKUPS=YES

cd \${SLURM_SUBMIT_DIR}/child-\${SLURM_ARRAY_TASK_ID}
. /work/ccm_gillespi/sw/reaxff/init_workdir.sh

time reac
rc=\$?
exit \$rc

EOF
#chmod u+x $INPUT.sh
#sbatch $INPUT.sh
chmod u+x submit_array.sh
sbatch submit_array.sh
#sleep 1 # pause to be kind to the scheduler
#done
# Go back to original directory
cd $scripts_dir
}


run_param_reduction() {
create_and_submit_param_reduction
number_remaining=$(squeue -u ${USER} | grep ${JOB_NAME} | wc -l)
while [ $number_remaining -gt ${JOB_THRESHOLD} ]; do
 echo "Number of remaining jobs: ${number_remaining}..."
 sleep ${JOB_INTERVAL}
 number_remaining=$(squeue -u ${USER} | grep ${JOB_NAME} | wc -l)
done
output_dir="00-param-reduction"
python3 -c "import param_reduction; param_reduction.read_and_write_param_reduction('${output_dir}')"
exit 0
}


#######################################
# RUNNING SCRIPT
#######################################
# Time
SECONDS=0

# Scheduled job
while [ $GENERATION_NUM -le $MAX_GENERATIONS ]; do
    main
	if [ "$USE_SLURM_ARRAY" = true ]; then
		submitRunsUsingArray
	else
		submitRunsWithoutArray
	fi

    number_remaining=${POP_SIZE}
    while [ $number_remaining -gt ${JOB_THRESHOLD} ]; do
        echo "Number of remaining jobs: ${number_remaining}..."
        sleep ${JOB_INTERVAL}
        number_remaining=$(squeue -u ${USER} | grep ${JOB_NAME} | wc -l)
    done 

    # Cancel remaining jobs 
    # scancel -u ${USER}  # INCOMPATIBLE WITH SLURM JOB
    #squeue -u ${USER} | grep 'GA' | awk '{print $1}' | xargs scancel

    GENERATION_NUM=$((GENERATION_NUM + 1))
    echo "Time Elapsed: ${SECONDS} seconds."
done

# To gather data for the very last generation
main

echo "Job Completed!"
echo "Time Elapsed: ${SECONDS} seconds."
