#!/usr/bin/env bash

DATADIR="${UVCGAN_DATA:-data}"

# Source: https://github.com/Onr/Council-GAN/blob/master/scripts/download.sh
declare -A URL_LIST=(
    [selfie2anime]="https://www.dropbox.com/s/9lz6gwwwyyxpdnn/selfie2anime.zip"
#    [male2female]="https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_male2female.zip"
#    [glasses]="https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_glasses.zip"
)

declare -A CHECKSUMS=(
    [selfie2anime]="2e8fe7563088971696d29af9f5153772733ac879c155c709b1aad741735ad7bc"
#    [male2female]="97178617b01af691b68f0b97de142c6be3331803b79906666fc9ab76f454a18e"
#    [glasses]="f4f141469fb8955822042d0999adcc81ec40db875c9bc930b733915b2089613f"
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

celeba_banner ()
{
    local dest="${1}"

    cat <<EOF
[NOTE] As of June 2023, the download links to the reference CouncilGAN's
CelebA datasets are no longer working. To use the CelebA datasets for I2I
translation, one needs to recreate them.

Please refer to https://github.com/LS4GAN/celeba4cyclegan for instructions
on how to do that.

Once the dataset is recreated, save it as

'${dest}'

to make it visible to 'uvcgan'.
EOF
}

exec_or_die ()
{
    "${@}" || die "Failed to execute: '${*}'"
}

calc_sha256_hash ()
{
    local path="${1}"
    sha256sum "${path}" | cut -d ' ' -f 1 | tr -d '\n'
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
        exec_or_die wget --no-check-certificate \
            "${url}" --output-document "${path}"
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

    download_archive  "${url}" "${zip}" "${checksum}"
    exec_or_die unzip "${DATADIR}/${zip}" -d "${DATADIR}"

    # exec_or_die rm "${dst}/${zip}"
}

check_dset_exists ()
{
    local path="${1}"

    if [[ -e "${path}" ]]
    then

        read -r -p "Dataset '${path}' exists. Overwrite? [yN]: " ret
        case "${ret}" in
            [Yy])
                exec_or_die rm -rf "${path}"
                ;;
            *)
                exit 0
                ;;
        esac
    fi
}

download_anime2selfie ()
{
    local url="${URL_LIST["selfie2anime"]}"
    local zip="selfie2anime.zip"
    local path="${DATADIR}/selfie2anime"

    check_dset_exists "${path}"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[selfie2anime]}"

    # CouncilGAN mangled dataset
    exec_or_die mv "${path}/trainA" "${path}/tmp"
    exec_or_die mv "${path}/trainB" "${path}/trainA"
    exec_or_die mv "${path}/tmp"    "${path}/trainB"

    exec_or_die mv "${path}/testA" "${path}/tmp"
    exec_or_die mv "${path}/testB" "${path}/testA"
    exec_or_die mv "${path}/tmp"   "${path}/testB"
}

download_male2female ()
{
    local path="${DATADIR}/celeba_male2female"
    celeba_banner "${path}"
}

download_glasses ()
{
    local path="${DATADIR}/celeba_glasses"
    celeba_banner "${path}"
}

move_files ()
{
    local dst="${1}"
    shift
    local src=( "${@}" )

    # NOTE: too many mv calls. Maybe optimize with xargs
    exec_or_die find "${src[@]}" -type f -exec mv '{}' "${dst}/" \;
}

download_celeba_all ()
{
    # NOTE: This dset is simply restructured male2female
    local path="${DATADIR}/celeba_all"
    local path_m2f="${DATADIR}/celeba_male2female"

    cat <<EOF
'celeba_all' is constructed from the 'celeba_male2female' dataset.

Please make sure that the 'celeba_male2female' is available by running
$ scripts/download_dataset.sh male2female

EOF
    check_dset_exists "${path}"

    [[ -e "${path_m2f}" ]] \
        || die "'celeba_male2female' is not found under '${path_m2f}'"

    exec_or_die mkdir -p "${path}/train" "${path}/val"

    echo "Copying files..."

    exec_or_die cp -r "${path_m2f}/trainA" "${path_m2f}/trainB" \
        "${path}/train/"

    exec_or_die cp -r "${path_m2f}/testA" "${path_m2f}/testB" \
        "${path}/val"
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

