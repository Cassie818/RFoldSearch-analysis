# RFoldseek subset database rebuild

Build date: 2026-09-16

Code: `DylanYang/RFoldseek`, branch `reproduce-shipped-db-tokens`, commit
`2305f07cb14236c2a17e988f798100096d35f5d5`.

The bundled checkpoint records `active_blocks = [g_ctx, g_nn1]`. Both databases
were encoded with those inactive descriptor blocks zeroed after normalization.

## Experimental subset

- Input: `/Users/cassie/Desktop/rfoldseek_rebuild_wip/nr80.jsonl`
- Clusters: `/Users/cassie/Desktop/rfoldseek_rebuild_wip/nr80.clstr`
- Output: `experimental/nr80_891_activeblocks.pt`
- Input records: 891
- Retained records (`min_len = 20`): 885
- CD-HIT clusters: 882
- Validation: all 885 token streams are byte-for-byte identical to
  `/Users/cassie/Desktop/RFoldSeek/files/models/db.pt`.
- SHA-256: `70f7e7eec018f4310cf77e23644c20f9f58be1a6df496e11bb6a855e9d03be03`

## Predicted subset

- Input: NuFold combined subset backbone JSONL (46,633 records)
- Source cluster file: `targets_nr80.fa.clstr`
- Output: `predicted/nufold_46633_activeblocks.pt`
- Retained records (`min_len = 20`): 46,633
- CD-HIT clusters: 10,333
- Input order preserved exactly; token values span all 80 states (0-79).
- SHA-256: `34a877889dbf7a63727880c774445e89db539ec5895c7d7cff5805221bdb2e7d`

The NuFold structure filenames end in `_rank_1`, whereas the CD-HIT identifiers
do not. `predicted/clusters_rank1_normalized.tsv` records the explicit mapping
used to assign all 46,633 entries; no singleton fallback was required.

These are subset validation databases. No database-specific E-value calibration
was fitted or embedded in either output. Refit calibration before treating an
E-value as a reported benchmark or deployment result.

## Experimental-to-predicted retrieval check

The 66 evaluable experimental queries were searched against the 46,633-chain
predicted subset with sequence-reachable pairs neutralized as in the existing
benchmark. At the deployed prefilter setting (`top_n = 500`, `min_diag = 2`,
tie seed 0), the rebuilt active-block pipeline obtained Recall@1/5/10 of
36/40/42 queries (0.545/0.606/0.636). The matched old-token control obtained
33/39/42 (0.500/0.591/0.636).

Among the 54 queries intersecting the saved sequence-blind cohort, Recall@1/5/10
increased from 31/36/39 (0.574/0.667/0.722) to 34/38/40
(0.630/0.704/0.741). Five queries became Recall@1 successes and two ceased to
be successes, for a net gain of three. These are subset results and are not a
replacement for the full 531,888-target benchmark.

- Active-block result: `experimental_to_predicted_top500_mindiag2.json`
- Old-token matched control: `control_oldtokens_experimental_to_predicted_top500_mindiag2.json`
- Sensitive active-block result (`top_n = 1000`, `min_diag = 1`):
  `experimental_to_predicted_top1000_mindiag1.json`

### Matched RiboSeek comparison

RiboSeek was rerun on the same 66 queries and targets using its native SA20
pipeline with a 500-candidate prefilter pool, tie seed 0, and the same neutral
pairs. RFoldseek versus RiboSeek Recall@1/5/10 was 36/40/42 versus 32/38/39.
On the 54-query sequence-blind intersection it was 34/38/40 versus 30/36/37.
At rank 1, 27 queries were successes for both methods, nine only for
RFoldseek, and five only for RiboSeek.

- RiboSeek matched result: `riboseek_experimental_to_predicted_top500.json`
