# Model and video artifacts

The source repository intentionally excludes checkpoints, MP4s, replay arrays, USD caches and W&B logs. A separate release bundle is prepared alongside the source bundle; it has not been uploaded to any GitHub repository yet.

Extract the release bundle at the repository root to obtain:

```text
artifacts/
  checkpoints/
    model_22000.pt                 # Quantitatively evaluated
    model_22000.contract.json
    model_25000.pt                 # Final training checkpoint; not yet evaluated
    model_25000.contract.json
  videos/
    duet_22000_locomotion.mp4       # 126 seconds
    duet_22000_height.mp4           # 72 seconds
    duet_22000_arms.mp4             # 54 seconds
  manifest.json                   # SHA256, size and checkpoint status
```

Contract JSON files replace the original host-local URDF path with a portable relative path. Their URDF hashes and control/observation metadata are unchanged. PyTorch weights are byte-for-byte copies of the trained checkpoints; do not load checkpoints from untrusted sources.

For publication, upload the source to your GitHub repository and the artifact archive as a GitHub Release attachment. Then add the actual Release URL to the README; no placeholder download endpoint or unrelated cloud account is used. Include the bundled license texts with redistributed artifacts.

The README preview GIF is included in Git. The three full MP4s remain in the release bundle. Checkpoint filenames use zero-based iteration labels: model22000 has22001 completed PPO updates and model25000 has25001.
