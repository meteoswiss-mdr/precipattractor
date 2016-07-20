#!/bin/bash
# Bash script to launch multiple jobs for computing radar rainfall statistics

outDir=/scratch/$USER/job_out # directory for printout
partition=postproc #postproc or normal

resourcesPerJob="--nodes=1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --mem-per-cpu=26g --time=2:00:00 \
--partition=$partition --account=msrad"

###### Arguments of python script ./radar_statistics.py
pyDir=/users/$USER/precipattractor/pyscripts # location of python script

wols=0 # Whether to use the weighted ordinary least squares
minR=0.08 # Minimum rainfall rate
accumMin=5 # Accumulation time of the product
fmt='netcdf'

###### Setting of separate time periods for multiproc
readarray periodsTimes < /users/$USER/precipattractor/shscripts/timePeriods.txt
nrPeriods=${#periodsTimes[@]}

maxNrCPUs = 120
if [ "$nrPeriods " -gt "$maxNrCPUs" ]; then
	echo "You asked for $nrPeriods CPUs"
	echo "You should reduce them to a maximum of $maxNrCPUs"
	exit 1
fi

## Looping over the different periods to launch separate jobs
for period in $(seq 0 $[$nrPeriods-1]); do
	periodLims=(${periodsTimes[$period]})
	startTime=${periodLims[0]}
	endTime=${periodLims[1]}
    
	#jobName=$(printf precip_attractor%03g $[$period+1])
	jobName=$(printf precip_attractor_%10i-%10i $startTime $endTime)
    
	#srun --job-name=$jobName --output=$outDir/$jobName.out --error=$outDir/$jobName.err $resourcesPerJob \
	#$pyDir/radar_statistics.py -start $startTime -end $endTime -wols $wols -minR $minR -accum $accumMin -format $fmt &
	
	echo "Job submitted:"
	echo "srun --job-name=$jobName --output=$outDir/$jobName.out --error=$outDir/$jobName.err $resourcesPerJob \
	#$pyDir/radar_statistics.py -start $startTime -end $endTime -wols $wols -minR $minR -accum $accumMin -format $fmt"
done


