#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=cbmm
#SBATCH -p cbmm
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --time=10:00:00
#SBATCH --output=output/%j.sh

DIRS_HIGH_LEVEL=($(ls -d -v ./results_new/*))
dir_path=${DIRS_HIGH_LEVEL[-1]}

echo "$dir_path"

'
lockfile="$dir_path/dir.lock"
tempfile=$(mktemp)

# Create a lock file to ensure mutual exclusion
(
    flock 200  # This will wait until the lock is available

    new_dir="$dir_path/0"

    # Check if the directory exists and is not empty
    if [ ! -d "$dir_path" ] || [ -z "$(ls -A "$dir_path")" ]; then
        echo "Directory does not exist or is empty."
    else
        DIR=($(find "$dir_path" -mindepth 1 -maxdepth 1 -type d | sort))

        found=0
        for idx in "${DIR[@]}"; do
            if [ ! -f "$idx/taken.txt" ]; then
                new_dir=$idx
                found=1
                break
            fi
        done

        if [ "$found" -eq 0 ]; then
            echo "No directory without taken.txt was found."
        else
            echo "First directory without taken.txt is: $new_dir"
        fi
    fi

    echo $new_dir 

    touch $new_dir/taken.txt

    # Write the new_dir value to the temporary file
    echo $new_dir > $tempfile

) 200>"$lockfile" # Use file descriptor 200 for the lock

# Read the new_dir value from the temporary file
new_dir=$(cat $tempfile)
rm -f $tempfile

# Ensure the new_dir variable is set correctly for running the Python script
if [ -z "$new_dir" ]; then
    echo "Error: new_dir variable is not set."
    exit 1
fi

#echo 'activating virtual environment'
#source ~/.bashrc
#conda activate ml
'
chmod u=rwx,g=r,o=r ./main.py
#module load openmind/gcc/11.1.0
#module load /om2/user/galanti/miniconda3/envs/pytorch/bin/python/3.9

echo 'running main.py'

#srun -n 1 python main.py --results-path=${new_dir} > ${new_dir}/print_${SLURM_PROCID}.txt 2>&1
python main.py --results-path="${new_dir}" #> "${new_dir}/print_$(echo $SLURM_PROCID).txt" 2>&1