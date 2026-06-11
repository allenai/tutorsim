To create these, we ran the v10 prompt through the annotation advisor and triaged the proposed changes as a human.

For rapport, we use --mode annotation for the advisor.
For scaffolding, we use --mode annotation_compare.

During scaffolding iteration (which occurred after rapport iteration), we modified the distribution of examples shown to the advisor so that it as based on each error type's proportional share of the limit budget based on its case count, rather than giving every type the full limit. That is, we used to show 20 examples per error type (e.g. pred: effective, true: partial), but now they are distributed based on error type frequency. 

Also, we removed unnecessarily keys from the output format to reduce cognitive load for the LM. 