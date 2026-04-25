# Evaluate Test Suite

This directory contains smoke-test launchers for evaluation code paths of:
- CATVTON
- OOTDiffusion
- IDMVTON
- StableVTON
- CPVTON
- custom DiT pretraining model

These tests intentionally use `--use_init_weights` so they do not require checkpoints.

Defaults used in tests:
- StreetTryOn split: `validation`
- Eval fractions: CurvTON `0.10`, Triplet `0.30`, StreetTryOn `0.30`
- Batch size per GPU/process: `8`
- Max batches: `1` (smoke test)

Dataset paths used:
- CurvTON test: `/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test`
- Triplet test: `/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1`
- StreetTryOn: `/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon`