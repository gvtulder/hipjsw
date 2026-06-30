# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

# Helper scripts for parameter selection based on the SLURM_ARRAY_TASK_ID.
#
# The including script should define a function define_parameters
# that calls the pick function for each of the parameters.
#
# Call the script as  $0 list  to list all experiments.


if [[ -z $SLURM_ARRAY_TASK_ID ]] ; then
  # debugging: count the number of experiments
  experiment_count=1

  pick() {
    local param_name=$1
    shift
    local choices=(${@})
    if [[ ${#choices[@]} == 1 ]] ; then s="" ; else s="s" ; fi
    echo "parameter $param_name: ${#choices[@]} choice$s"
    experiment_count=$(( $experiment_count * ${#choices[@]} ))
  }

  define_parameters

  if [[ $experiment_count == 1 ]] ; then s="" ; else s="s" ; fi
  echo "There are $experiment_count experiment$s."

  if [[ $1 == list ]] || [[ $1 == demo ]] ; then
    # list all experiments
    pick() {
      # picks a parameter choice based on the $array_task_id
      local param_name=$1
      shift
      local choices=(${@})
      local idx=$(( $array_task_id % ${#choices[@]} ))
      array_task_id=$(( $array_task_id / ${#choices[@]} ))
      echo -ne "\t${param_name}=${choices[$idx]}"
    }
    for array_task_id in $(seq 1 $experiment_count ) ; do
      echo -n "${array_task_id}:"
      array_task_id=$(( $array_task_id - 1 ))
      define_parameters
      echo
    done
  else
    echo "Run $0 list to list them all."
  fi

  exit 1

else
  # for real: select an experiment based on $SLURM_ARRAY_TASK_ID
  array_task_id=$(( $SLURM_ARRAY_TASK_ID - 1))
  param_list=""

  pick() {
    # picks a parameter choice based on the $array_task_id
    local param_name=$1
    shift
    local choices=(${@})
    local idx=$(( $array_task_id % ${#choices[@]} ))
    array_task_id=$(( $array_task_id / ${#choices[@]} ))
    echo "param: ${param_name} = ${choices[$idx]}"
    declare -g ${param_name}=${choices[$idx]}
    if [[ ! -z "${param_list}" ]] ; then
      param_list="${param_list}-"
    fi
    param_list="${param_list}${param_name}=${choices[$idx]}"
  }

  define_parameters
fi

