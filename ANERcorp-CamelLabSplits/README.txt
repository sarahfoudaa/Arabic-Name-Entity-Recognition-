********************************************************************
ANERcorp -- CAMeL Lab  Train/Test Splits
********************************************************************

Since its creation in 2008, the ANERcorp dataset (Benajiba & Rosso,
2008) has been a standard reference used by Arabic named entity
recognition researchers around the world. However, over time, this
dataset was copied over from user to user, modified slightly here and
there, and split in many different configurations that made it hard
to compare fairly across papers and systems.

In 2020, a group of researchers from CAMeL Lab (Habash, Alhafni and
Oudah), and Mind Lab (Antoun and Baly) met with the creator of the
corpus, Yassine Benajiba, to consult with him and collectively agree
on an exact split, and accepted minor corrections from the original
dataset. Bashar Alhafni from CAMeL Lab working with Nizar Habash
implemented the decisions provided in this release.

Changes from the original dataset include the following:

1) We corrected minor tag spelling errors.

2) We converted middle periods (·) and bold periods (•) to regular periods (.).

3) We removed the blank unicode character (\u200F).

4) We added sentence boundaries after sequences of 1 or more periods.

5) We split the dataset sequentially. The sentences containing the first
5/6 of the words go to train and the rest go to test. The train split
has 125,102 words and the test split has 25,008 words.

If you use this corpus split, please cite (a) the original ANERcorp
paper and (b) the splits as specified in the CAMeLTools paper:

Benajiba, Yassine, Paolo Rosso, and José Miguel Benedí Ruiz. "Anersys:
An Arabic named entity recognition system based on maximum entropy."
In International Conference on Intelligent Text Processing and
Computational Linguistics, pp. 143-153. Springer, Berlin, Heidelberg,
2007.
https://link.springer.com/chapter/10.1007/978-3-540-70939-8_13

Ossama Obeid, Nasser Zalmout, Salam Khalifa, Dima Taji, Mai Oudah,
Bashar Alhafni, Go Inoue, Fadhl Eryani, Alexander Erdmann, and Nizar
Habash. "CAMeL Tools: An Open Source Python Toolkit, for Arabic Natural
Language Processing." In  Proceedings of the Conference on Language
Resources and Evaluation (LREC 2020), Marseille, 2020.
https://www.aclweb.org/anthology/2020.lrec-1.868.pdf
