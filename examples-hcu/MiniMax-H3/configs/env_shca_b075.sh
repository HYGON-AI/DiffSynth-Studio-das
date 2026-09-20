#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Source inside HCU framework container after its login environment.
# Verified with 16 BF16 collective ranks on HCU cluster.

export LD_LIBRARY_PATH="/opt/rccl-rdma-sharp-plugins/lib:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export NCCL_TOPO_FILE=/usr/local/built-in-508-topo-input-tj-default.xml
export NCCL_PXN_DISABLE=0
export RCCL_PXN_GPU_BALANCE=1
export RCCL_NET_PLANE='shca_0,shca_3|shca_1,shca_2'
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export HIP_VISIBLE_DEVICES=0,1,5,4,2,3,7,6
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DELAY_UNLINK_SHM=1
# Keep INFO during initial training validation; set NCCL_DEBUG=WARN later.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

if [[ ! -r /opt/rccl-rdma-sharp-plugins/lib/librccl-net-shca.so || ! -r "${NCCL_TOPO_FILE}" ]]; then
  echo "Missing SHCA plugin/topology; install the cluster-provided plugin and topology on every node." >&2
  return 1
fi
