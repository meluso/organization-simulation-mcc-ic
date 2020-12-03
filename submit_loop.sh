for i in $(seq 0 99)
do
	sbatch --export=ii=${i} submit_exec002.sbat
done