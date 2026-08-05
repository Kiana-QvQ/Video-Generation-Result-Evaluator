# Data Policy

Large videos, audio, archives, and model checkpoints should not be added to
the source repository as ordinary Git blobs.

- Keep source code and small manifests in Git.
- Store large media/checkpoints with Git LFS, DVC, or object storage.
- Record source URL, license, file size, SHA256, and extraction version in a
  manifest.
- Run `python scripts/audit_dataset.py` after restoring a dataset.
- Use `--strict` in CI or before AU profile/classifier training.

The existing checked-in data is intentionally not deleted by this change.
Future large files are covered by `.gitattributes`; migrating historical blobs
requires an explicit repository-history operation.
