# Upload the prepared repository to GitHub

The local Git repository contains code and metadata only. The two `.pth` files exist in `models/`
but are ignored by Git. Do not use `git add -f` for those files and do not enable Git LFS for this release.

## Repository

1. Create an empty repository named `bal-cytospin-pipeline` under your intended GitHub account or lab organisation.
   Choose visibility and the project code/weights licence. Keep the generated repository empty: no additional
   README or initial commit, because this local preparation already has its own initial commit.
2. From this prepared folder, set the actual remote and push:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/bal-cytospin-pipeline.git
git push -u origin main
git push origin thesis-v1.0
```

## Weights release

Open **Releases > Draft a new release**, select the existing `thesis-v1.0` tag and attach these three local files:

- `models/dinobloom_bal_production.pth`
- `models/neutrophil_expert_finetuned.pth`
- `models/SHA256SUMS.txt`

Use `docs/RELEASE_NOTES.md` for the description. Both weights are below GitHub's 2 GiB per-asset limit.
Publish after reviewing the owner, visibility, description and licences. Creating/publishing a remote release
has not been done by the preparation process.

Once the repository is public, update the README with its actual URL. The downloader accepts `--repo OWNER/REPO`;
it does not require a hard-coded account. For private repositories, download assets in an authenticated browser
and place them in `models/`; the public-release downloader does not handle private asset authentication.

Verify the files before upload:

```bash
python3 scripts/verify_models.py
```

Release attachments are separate from the automatically generated GitHub source ZIP. Users need to download
the two weights explicitly or run the downloader after cloning. Link the eventual Zenodo archive from the README.
See [GitHub's release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).
