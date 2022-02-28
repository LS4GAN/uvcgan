#!/usr/bin/env bash

DATADIR="${UVCGAN_DATA:-data}"

# Source: https://github.com/Onr/Council-GAN/blob/master/scripts/download.sh
declare -A URL_LIST=(
    [selfie2anime]="https://www.dropbox.com/s/9lz6gwwwyyxpdnn/selfie2anime.zip"
    [male2female]="https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_male2female.zip"
    [glasses]="https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_glasses.zip"
)

declare -A CHECKSUMS=(
    [selfie2anime]="c2a79b0bccc640042413adf3113a384415058ec4065c2fd8aff646f08af12b86"
    [male2female]="97178617b01af691b68f0b97de142c6be3331803b79906666fc9ab76f454a18e"
    [glasses]="f4f141469fb8955822042d0999adcc81ec40db875c9bc930b733915b2089613f"
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_dataset.sh DATASET

where DATADIR is one of selfie2anime, male2female, glasses, or celeba_all.
EOF

    if [[ $# -gt 0 ]]
    then
        die "${*}"
    else
        exit 0
    fi
}

exec_or_die ()
{
    "${@}" || die "Failed to execute: '${*}'"
}

calc_sha256_hash ()
{
    local path="${1}"
    sha256sum "${path}" | cut  -d ' ' -f | tr -d '\n'
}

download_archive ()
{
    local url="${1}"
    local archive="${2}"
    local checksum="${3}"

    exec_or_die mkdir -p "${DATADIR}"

    local path="${DATADIR}/${archive}"

    if [[ ! -e "${DATADIR}/${archive}" ]]
    then
        exec_or_die wget "${url}" --output-document "${path}"
    fi

    if [[ -n "${checksum}" ]]
    then
        local test_csum="$(calc_sha256_hash "${path}")"

        [[ "${test_csum}" == "${checksum}" ]] || \
            die "Checksum mismatch for '${path}' ${test_csum} vs ${checksum}"

    fi
}

download_and_extract_zip ()
{
    local url="${1}"
    local zip="${2}"
    local checksum="${3}"

    download_archive  "${url}" "${zip}"
    exec_or_die unzip "${DATADIR}/${zip}" -d "${DATADIR}"

    # exec_or_die rm "${dst}/${zip}"
}

download_anime2selfie ()
{
    local url="${URL_LIST["selfie2anime"]}"
    local zip="selfie2anime.zip"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[selfie2anime]}"
}

download_male2female ()
{
    local url="${URL_LIST["male2female"]}"
    local zip="male2female.zip"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[male2female]}"
}

move_files ()
{
    local dst="${1}"
    shift
    local src=( "${@}" )

    # NOTE: too many mv calls. Maybe optimize with xargs
    exec_or_die find "${src[@]}" -type f -exec mv '{}' "${dst}/" \;
}

download_glasses ()
{
    local url="${URL_LIST["glasses"]}"
    local zip="glasses.zip"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[glasses]}"

    local dset_dir="${DATADIR}/glasses"

    for subdir in {trainA,trainB,testA,testB}
    do
        echo "Restructuring directory: '${subdir}'"
        for splitdir in {1,2}
        do
            local src="${dset_dir}/${subdir}/${splitdir}"
            local dst="${dset_dir}/${subdir}/"

            move_files "${dst}" "${src}"
            exec_or_die rmdir "${src}"
        done
    done

    exec_or_die mv "${dset_dir}" "${DATADIR}/celeba_glasses"
}

download_celeba_all ()
{
    # NOTE: This dset is simply restructured male2female
    local url="${URL_LIST["male2female"]}"
    local zip="male2female.zip"
    local dset_dir="${DATADIR}/celeba_all"

    download_archive "${url}" "${zip}" "${CHECKSUMS[male2female]}"
    exec_or_die unzip "${DATADIR}/${zip}" -d "${dset_dir}"

    local unzipped_path="${dset_dir}/celeba_male2female"

    exec_or_die mkdir -p "${dset_dir}/train" "${dset_dir}/test"

    move_files "${dset_dir}/train" \
        "${unzipped_path}/trainA" "${unzipped_path}/trainB"

    move_files "${dset_dir}/test" \
        "${unzipped_path}/testA" "${unzipped_path}/testB"

    for subdir in {trainA,trainB,testA,testB}
    do
        exec_or_die rmdir "${unzipped_path}/${subdir}"
    done

    exec_or_die rmdir "${unzipped_path}"
}

dataset="${1}"

case "${dataset}" in
    selfie2anime|anime2selfie)
        download_anime2selfie
        ;;
    male2female|celeba_male2female_preproc)
        download_male2female
        ;;
    glasses|eyeglasses|celeba_glasses_preproc)
        download_glasses
        ;;
    celeba_all|celeba_preproc)
        download_celeba_all
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage "Unknown dataset '${dataset}'"
esac

