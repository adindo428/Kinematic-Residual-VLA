# Workspace backup — November 3, 2026

This snapshot preserves local source, experiment scripts, paper files, run records, evidence, and historical transfer archives. Remote-only datasets, simulator assets, environments, and training checkpoints are not backed up by this commit. Local caches and machine-specific environment files are excluded. See manifest.json for hashes and scope.

The original 149,972,223-byte transfer archive exceeds GitHub’s file limit and is stored losslessly in ordered parts. Reconstruct it from the repository root:

```sh
cat backup/github_20261103/palmetto-prepared-family.tar.gz.part* > palmetto-prepared-family.tar.gz
shasum -a 256 palmetto-prepared-family.tar.gz
```

Compare the checksum with the original archive entry in manifest.json. Git ignores many historical backup files by default; they were explicitly included in this snapshot. Future untracked maintenance records and archives need explicit staging.
