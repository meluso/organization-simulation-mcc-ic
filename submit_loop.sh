for i in $(seq 0 499)
do
	sbatch --export=ii=${i} submit_exec003.sbat
done