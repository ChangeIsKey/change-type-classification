# Using Synchronic Definitions and Semantic Relations to Classify Semantic Change Types
This repository contains the code accompanying the paper titled "Using Synchronic Definitions and Semantic Relations to Classify Semantic Change Types"

## Paper Abstract
There is abundant evidence of the fact that the way words change their meaning can be classified in different _types_ of change, highlighting the relationship between the old and new meanings (among which generalization, specialization and co-hyponymy transfer).
In this paper, we present a way of detecting these types of change by constructing a model that leverages information both from synchronic lexical relations and definitions of word meanings. Specifically, we use synset definitions and hierarchy information from WordNet and test it on a digitized version of Blank's (1997) dataset of semantic change types. Finally, we show how the sense relationships can improve models for both approximation of human judgments of semantic relatedness as well as binary Lexical Semantic Change Detection.

Snippet of the LSC Cause-Type-Definitions Benchmark:
![alt text](https://github.com/ChangeIsKey/change-type-classification/blob/main/lsc_ctd_benchmark_snippet_table.png "t")

<b> Citation </b>

```
@inproceedings{cassotti-etal-2024-using,
    title = "Using Synchronic Definitions and Semantic Relations to Classify Semantic Change Types",
    author = "Cassotti, Pierluigi  and
      De Pascale, Stefano  and
      Tahmasebi, Nina",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.249",
    pages = "4539--4553",
    abstract = "There is abundant evidence of the fact that the way words change their meaning can be classified in different types of change, highlighting the relationship between the old and new meanings (among which generalisation, specialisation and co-hyponymy transfer).In this paper, we present a way of detecting these types of change by constructing a model that leverages information both from synchronic lexical relations and definitions of word meanings. Specifically, we use synset definitions and hierarchy information from WordNet and test it on a digitized version of Blank{'}s (1997) dataset of semantic change types. Finally, we show how the sense relationships can improve models for both approximation of human judgments of semantic relatedness as well as binary Lexical Semantic Change Detection.",
}
```

## Model
The model is available at [https://huggingface.co/ChangeIsKey/change-type-classifier](https://huggingface.co/ChangeIsKey/change-type-classifier)

### LSC-CTD Benchmark
The LSC-CTD (Lexical Semantic Change Cause-Type-Definitions) Benchmark is available on Zenodo:
[https://zenodo.org/records/11471318](https://zenodo.org/records/11471318)

## Contents
This repository includes the following files:

`src`: Python code.

