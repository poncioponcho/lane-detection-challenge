#!/usr/bin/env bash
# split_upload.sh <local_file> <remote_file> — 200m 分块、逐块 md5+重试、整文件终验。
# 分块名嵌入源文件标识 → 多个上传可安全并行。
# 用法: bash split_upload.sh <local> <remote>   （或 source 后调用 split_upload）
# 环境变量: AB_SSH_PORT（默认 59725）、AB_STAGING（默认 /hy-tmp/datasets/_staging）
set -uo pipefail
PORT="${AB_SSH_PORT:-59725}"
SSH="ssh -i $HOME/.ssh/lane_id -p $PORT -o BatchMode=yes -o ConnectTimeout=20 root@i-1.gpushare.com"
SCP="scp -i $HOME/.ssh/lane_id -P $PORT -o BatchMode=yes -o ConnectTimeout=20"
STAGING="${AB_STAGING:-/hy-tmp/datasets/_staging}"

split_upload() {
    local src=$1 dst=$2
    local safebase=$(basename "$src" | tr -c 'A-Za-z0-9._-' '_')
    local tag="schunk_${safebase}."
    local lmd5=$(md5 -q "$src")
    echo "  split_upload $safebase ($(du -h "$src" | cut -f1)) md5=$lmd5"
    local whole=$($SSH "md5sum $dst 2>/dev/null" | awk '{print $1}')
    if [ "$whole" = "$lmd5" ]; then
        echo "  $dst already verified on remote (whole-file skip)"
        return 0
    fi
    rm -f /tmp/${tag}*
    split -b 200m "$src" "/tmp/${tag}"
    local chunks=(/tmp/${tag}*)
    echo "  ${#chunks[@]} chunks; resume-check remote existing..."
    local todo=()
    local c bn l md5r attempt
    for c in "${chunks[@]}"; do
        bn=$(basename "$c")
        md5r=$($SSH "md5sum $STAGING/$bn 2>/dev/null" | awk '{print $1}')
        l=$(md5 -q "$c")
        if [ "$l" = "$md5r" ]; then
            echo "    chunk $bn already on remote (skip)"
        else
            todo+=("$c")
        fi
    done
    echo "  to upload: ${#todo[@]} chunks"
    for c in ${todo[@]+"${todo[@]}"}; do
        bn=$(basename "$c")
        for attempt in 1 2 3 4 5 6; do
            $SCP -q "$c" "root@i-1.gpushare.com:$STAGING/$bn" && break
            echo "    chunk $bn attempt $attempt failed; retry in 20s"; sleep 20
        done
        l=$(md5 -q "$c"); md5r=$($SSH "md5sum $STAGING/$bn" | awk '{print $1}')
        [ "$l" = "$md5r" ] || { echo "    chunk $bn md5 mismatch"; return 1; }
        echo "    chunk $bn OK"
    done
    $SSH "cat $STAGING/${tag}* > $dst && rm -f $STAGING/${tag}*"
    md5r=$($SSH "md5sum $dst" | awk '{print $1}')
    [ "$lmd5" = "$md5r" ] || { echo "  whole-file md5 mismatch"; return 1; }
    echo "  split_upload $safebase DONE"
    rm -f /tmp/${tag}*
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    split_upload "${1:?local}" "${2:?remote}"
fi
