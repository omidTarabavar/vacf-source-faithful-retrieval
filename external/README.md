# External repositories

This cleaned release intentionally does not redistribute third-party source
repositories or nested `.git` histories.

## AVerImaTeC organizer repository

- Repository: https://github.com/abril4416/AVerImaTec_Shared_Task.git
- Commit used in the archived workspace: `f6f361d2fc5a0ab7d3ea282e2beced56b928e1bd`

The VACF evaluation scripts import organizer evaluation utilities from this
repository. Clone it separately and use the recorded commit when possible.

## XxP comparator repository

- Repository: https://github.com/XplaiNLP/FEVER-9-AVerImaTeC-XxP.git
- Commit found in the archived workspace: `c682eb8cd6be1cd7fce241d71e5e9c6a42657537`

The matched comparison/downstream scripts rely on modules from this repository
(e.g., `GenerationLLMHandler`). Clone it separately rather than copying its
source into this repository.

## Why third-party code is not included

The raw export contained complete third-party working trees and nested Git
metadata. Those were useful for internal provenance, but are inappropriate for
a clean public research-code repository. The public release therefore records
URLs and commits instead.
