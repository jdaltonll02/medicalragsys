Data from the BioASQ Synergy task in the context of BioASQ13 (2025).

- The files named testset_x.json are the files of the testsets released for each round (x) of the task. 
- The files named feedback_accompanying_round_x.json are the files with feedback based on the submissions in previous rounds, which were released with the testset file for each round x.
- The files named golden_round_x.json are the files used for the final evaluation of systems responses in each round. For document and snippet retrieval, documents and snippets from previous feedback files are excluded.

The numbering of the files (values of x) ranges from 1 to 4 for the four rounds of this version of the task. Still, the field "answerReady_on.round" ranges from 1 to 8, considering rounds from the previous version of the Synergy task (BioASQ11, 2023). That is, a question that turned into "ready to answer" in the testset_round_1.json, would have the value "5" for the "answerReady_on.round" field.