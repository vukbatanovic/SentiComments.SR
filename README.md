# SentiComments.SR - A Sentiment Analysis Dataset of Comments in Serbian
The SentiComments.SR dataset includes the following three corpora:
- The main SentiComments.SR corpus, consisting of 3490 movie-related comments.
- The movie verification corpus, consisting of 464 movie-related comments.
- The book verification corpus, consisting of 173 book-related comments.

## Dataset creation
The main SentiComments.SR corpus was constructed out of the comments written by visitors on the [kakavfilm.com](http://kakavfilm.com) movie review website in Serbian.
Each comment was assigned a unique ID based on the movie to which it referred, and its placement in the comment tree for that particular movie, signifying its position within the whole discussion.
Comments containing more than a predefined upper bound for token count (using basic whitespace tokenization), were discarded, as were the comments not written in Serbian.

The movie verification corpus comments were sourced from two other Serbian movie review websites - gledajme.rs and [happynovisad.com](http://www.happynovisad.com).
The book verification corpus comments were also sourced from the [happynovisad.com](http://www.happynovisad.com) website.

## Dataset annotation
Six sentiment labels were used in dataset annotation: +1, -1, +M, -M, +NS, and -NS, with the addition of an 's' suffix to labels denoting the sentiment of a sarcastic text.
The annotation principles used to assign sentiment labels to items in SentiComments.SR are described in the papers listed in the References section.

The main SentiComments.SR corpus was annotated by two annotators working together, and therefore contains a single, unified sentiment label for each comment.
The verification corpora were used to evaluate the quality, efficiency, and cost-effectiveness of the annotation framework, which is why they contain separate sentiment labels for six annotators, divided into three groups - initial group (#1 and #2), experimental group (#3 and #4) and control group (#5 and #6).

## Dataset format
The main SentiComments.SR corpus is available as a tab-separated .txt file, in two variants:
* [SentiComments.SR - original](./SentiComments.SR.orig.txt) - the original, unaltered comment texts
* [SentiComments.SR - corrected](./SentiComments.SR.corr.txt) - the corrected, manually proofed comment texts

Both variants have the following column structure, and they both share the same sentiment labels and comment IDs:
* Column 1 - Sentiment label
* Column 2 - Comment ID
* Column 3 - Comment text

The verification corpora are also available as tab-separated .txt files:
* [The movie verification corpus](./SentiComments.SR.verif.movies.txt)
* [The book verification corpus](./SentiComments.SR.verif.books.txt)

The column structure of the verification corpora files is as follows:
* Columns 1-6 - Sentiment labels assigned by annotators 1-6
* Column 7 - Comment text

Comments in all files are written in either the Serbian Latin or the Serbian Cyrillic script.
All files are encoded in UTF-8.

## References
If you wish to use the SentiComments.SR dataset (or the annotation principles applied in its construction) in your paper or project, please cite the following paper:

* **[A versatile framework for resource-limited sentiment articulation, annotation, and analysis of short texts](https://doi.org/10.1371/journal.pone.0242050)**, Vuk Batanović, Miloš Cvetanović, Boško Nikolić, PLoS ONE, 15(11): e0242050 (2020).

The reference for the full annotation guidelines (in Serbian) is the following:

* **[A methodology for solving semantic tasks in the processing of short texts written in natural languages with limited resources](https://nardus.mpn.gov.rs/handle/123456789/17783)**, Vuk Batanović, PhD thesis, University of Belgrade - School of Electrical Engineering (2020).

## Acknowledgement
The annotation of the SentiComments.SR dataset was supported by the [Regional Linguistic Data Initiative](http://reldi.spur.uzh.ch/) (*ReLDI*) via the Swiss National Science Foundation grant no. 160501.

## License
The SentiComments.SR dataset is available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/) license. The entire text of the license can be found [here](http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

If you wish to use this dataset for a commercial product, please contact me at: vuk.batanovic / at / ic.etf.bg.ac.rs
