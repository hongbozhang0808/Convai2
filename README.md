# Convai2
Projects for Convai2
=================================================

Dependencies:
-------
  * python ( 3.5.2 )
  * keras ( 2.1.5 )
  * tensorflow ( 1.3.0 )
  * keras_attention_block ( 0.0.2 ）
  * gensim（ 3.2.0 ）
  * nltk（ 3.2.1 ）

Team info
-------
  * team name: team PAT
  * model name: topicSeq2seq

Evaluation files
-------
  * projects/convai2/s2s_vae/eval_f1.py (15.54 on valid_self_revised_no_cands.txt)
  * projects/convai2/s2s_vae/eval_ppl.py (50.37 on valid_self_revised_no_cands.txt)
 
Model files
-------
  * the pretrained model files are in data/models/convai2/, including three models.
  * the compressed files in data/models/convai2/seq2seq-model/* must be uncompressed at first, and the uncompressed file should be constructed as data/models/convai2/seq2seq-model/weights_s2s_v1_3_7_5.hdf5
  
Modified files
--------
  * these files have been modified, including:
  * Convai2/projects/convai2/eval_ppl.py
  * Convai2/parlai/tasks/convai2/agents.py
  * Convai2/projects/convai2/eval_f1.py
