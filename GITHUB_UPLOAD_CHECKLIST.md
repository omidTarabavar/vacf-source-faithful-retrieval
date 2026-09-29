# GitHub / Zenodo release checklist

1. Create a public GitHub repository named `vacf-source-faithful-retrieval`.
2. Upload the contents of this folder to the repository root (not the enclosing folder itself).
3. Review `LICENSE_NOTICE.md` and choose a license for author-owned code after co-author/institution approval.
4. Do not add API keys, model caches, benchmark data archives, or nested third-party repositories.
5. Add the final claim-level derived artifacts listed in `docs/MISSING_ARTIFACTS.md` if available. Large artifacts can be placed in Zenodo instead of GitHub.
6. Create a GitHub release (suggested tag: `v1.0.0-paper`).
7. Connect the repository/release to Zenodo and mint a DOI for a versioned archival release.
8. Replace `[GITHUB_URL]` and `[ZENODO_DOI_OR_URL]` in the manuscript Data Availability wording.
9. Add the paper DOI/citation to `CITATION.cff` after publication.
10. Re-run `python check_release.py` before the release.
