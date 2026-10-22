# Changelog

## Deployment tools — 2026-10-23

- Add normalized TorchScript export, MuJoCo physics deployment and an experimental G1 EDU 23DoF HG DDS controller.
- Add observation/SDK mapping checks, operator-controlled startup, latched watchdog stops and offline fault tests.
- Record preliminary 22000-checkpoint MuJoCo results; hardware validation remains pending.

## 0.1.0 — portfolio snapshot, 2026-10-20

- Package the G1 23DoF IsaacLab locomotion and arm-disturbance task only; keep inherited robot/contact support under common without registering another task.
- Replace machine-specific asset paths with portable overrides and a checked asset downloader.
- Add the policy viewer, generic checkpoint labels in replay, public documentation and source CI.
- Preserve22000-checkpoint evaluation and25001-update training completion as distinct evidence.
- No intentional reward, curriculum, joint-gain or network changes in publication cleanup.
