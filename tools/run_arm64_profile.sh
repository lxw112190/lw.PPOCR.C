#!/usr/bin/env bash
set -euo pipefail

root_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_dir="${root_dir}/results"
if [[ "${1:-}" == "--output" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "usage: $0 [--output directory]" >&2
        exit 2
    fi
    output_dir=$2
    shift 2
fi
if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--output directory]" >&2
    exit 2
fi
if [[ "${output_dir}" != /* ]]; then
    output_dir="${root_dir}/${output_dir}"
fi

python3 "${root_dir}/tools/collect_native_ocr_profile.py" \
    --profile-driver "${root_dir}/bin/full-ocr-profile-driver" \
    --benchmark-driver "${root_dir}/bin/full-ocr-intra-benchmark" \
    --det "${root_dir}/models/det.lwm" \
    --cls "${root_dir}/models/cls.lwm" \
    --rec "${root_dir}/models/rec.lwm" \
    --dictionary "${root_dir}/models/ppocr_keys.txt" \
    --image "${root_dir}/models/sample.ppm" \
    --variant tiny \
    --target-width 960 \
    --expected-backend neon \
    --require-rss \
    --output "${output_dir}"
