#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setenv.sh"

usage() {
    cat <<'EOF'
Usage:
  dacpp.sh translate <source.dac.cpp> [translator-options] [-- compiler-options]
  dacpp.sh build <generated.cpp> [output-bin]
  dacpp.sh translate-build <source.dac.cpp> [output-bin] [translator-options]

Common translator options:
  --mode=buffer        Generate SYCL buffer code.
  --mpi                Enable MPI code generation.

Environment:
  ACPP_ROOT or ADAPTIVECPP_ROOT   AdaptiveCpp install prefix for build commands.
  DACPP_SYCL_COMPILER             auto, adaptivecpp, icpx, or mpicxx.
  ONEAPI_SETVARS/ONEAPI_ROOT      Optional oneAPI setup for icpx.
  ICPX / MPICXX                   Explicit compiler paths.
  DACPP_TRANSLATOR                Path to the translator executable.
  DACPP_TMP_ROOT                  Default output directory for build artifacts.
EOF
}

require_args() {
    local command="$1"
    local count="$2"
    if [[ "$count" -lt 1 ]]; then
        echo "dacpp.sh: missing input for '$command'" >&2
        usage >&2
        exit 2
    fi
}

command="${1:-}"
if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
    usage
    exit 0
fi
shift

case "$command" in
    translate)
        require_args "$command" "$#"
        dacpp "$@"
        ;;

    build|compile)
        require_args "$command" "$#"
        dacpp-compile "$@"
        ;;

    translate-build)
        require_args "$command" "$#"
        source_file="$1"
        shift

        output_bin=""
        if [[ $# -gt 0 && "$1" != --* ]]; then
            output_bin="$1"
            shift
        fi

        dacpp "$source_file" --mode=buffer "$@"
        generated_cpp="$(generated_cpp_for "$source_file")"

        if [[ -n "$output_bin" ]]; then
            dacpp-compile "$generated_cpp" "$output_bin"
        else
            dacpp-compile "$generated_cpp"
        fi
        ;;

    *)
        echo "dacpp.sh: unknown command '$command'" >&2
        usage >&2
        exit 2
        ;;
esac
