#!/bin/bash

export LC_ALL=C  

export OMP_PLACES=threads

declare -r PROGRAM=${0##*/}
declare -r ITERATION_COUNT=1000

if [ -z "$1" ]; then
  APPLICATION="ones-stream-full"
else
  APPLICATION="$1"
fi

if [ "$APPLICATION" == *"ones-npb-ep"* ]; then
  declare -r PROBLEM_SIZE=22
elif [ "$APPLICATION" == *"ones-solvers-cg"* ]; then
  declare -r PROBLEM_SIZE=5000
elif [ "$APPLICATION" == *"ones-solvers-bicgstab"* ]; then
  declare -r PROBLEM_SIZE=5000
else
  declare -r PROBLEM_SIZE=33554432
fi


declare -r DATADIR=./experiment_inputs/identification_inputs
declare -r OUTPUTDIR=./experiment_data/${APPLICATION}_identification
declare -r BENCHMARK='stream_c'
declare -r RUNNER='identification'
declare -r PARAMS_FILE='parameters.yaml'

declare -r TOPOLOGY_FILE='topology.xml'

if [ ! -d "$OUTPUTDIR" ]; then
	mkdir -p "$OUTPUTDIR"
fi




declare -ra PRERUN_SNAPSHOT_FILES=(
        "${PARAMS_FILE}"
        "${TOPOLOGY_FILE}"
)

declare -ra SYSTEM_STATE_SNAPSHOT_FILES=(
        /proc/cpuinfo
        /proc/iomem
        /proc/loadavg
        /proc/meminfo
        /proc/modules
        /proc/stat
        /proc/sys/kernel/hostname
        /proc/uptime
        /proc/version
        /proc/vmstat
        /proc/zoneinfo
)

declare -a POSTRUN_SNAPSHOT_FILES=(
        dump_pubMeasurements.csv
        dump_pubProgress.csv
	    identification-runner.log
)


function dump_parameters {
	declare -r timestamp="${1}"
        declare -r runner="${2}"
        declare -r cfg="${3}"
        declare -r benchmark="${4}"
        declare -r extra="${*:5}"
	cat <<- EOF > "${OUTPUTDIR}/${PARAMS_FILE}"
		timestamp: ${timestamp}
		runner: ${runner}
		config-file: ${cfg##*/}
		benchmark: ${BENCHMARK}
		extra: ${extra}
	EOF
}


function snapshot_system_state {
        archive="${1}"
        subdir="${2}"

        wd=$(mktemp --directory)
        mkdir "${wd}/${subdir}"

        for pseudofile in "${SYSTEM_STATE_SNAPSHOT_FILES[@]}"; do
                saveas="$(basename "${pseudofile}")"
                cat "${pseudofile}" > "${wd}/${subdir}/${saveas}"
        done

        tar --append --file="${archive}" --directory="${wd}" --transform='s,^,sysstate/,' -- "${subdir}"

        rm --recursive --force -- "${wd}"
}

for cfg in "$DATADIR"/*
do
	if [[ ${cfg} == *"step"* ]]; then
		timestamp="$(date --iso-8601=seconds)"
		archive="${OUTPUTDIR}/preliminaries_${BENCHMARK}_${timestamp}.tar"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${cfg}"
		dump_parameters "${timestamp}" "${RUNNER}" "${cfg}" "${BENCHMARK}" "--iterationCount=${ITERATION_COUNT} --problemSize=${PROBLEM_SIZE}"
		lstopo --output-format xml --whole-system --force "${OUTPUTDIR}/${TOPOLOGY_FILE}"
		tar --create --file="${archive}" --files-from=/dev/null
		tar --append --file="${archive}" --transform='s,^.*/,,' -- "${cfg}"
		tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${PRERUN_SNAPSHOT_FILES[@]}"
		snapshot_system_state "${archive}" 'pre'
		echo $APPLICATION
        if [ "$APPLICATION" == *"ones-solvers-cg"* ]; then
		  python identification.py ${cfg} ones-solvers-cg 10000 poor 0 10000
        elif [ "$APPLICATION" == *"ones-solvers-bicgstab"* ]; then
		  python identification.py ${cfg} ones-solvers-bicgstab 20000 poor 0 10000
        else
          python identification.py ${cfg} -- $APPLICATION ${PROBLEM_SIZE} ${ITERATION_COUNT}
        fi
		tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${POSTRUN_SNAPSHOT_FILES[@]}"
		snapshot_system_state "${archive}" 'post'
		xz --compress "${archive}"
		echo __________________________________________________________________________________________________
	fi

done

