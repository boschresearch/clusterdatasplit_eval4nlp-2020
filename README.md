# ClusterDataSplit

This repository holds the companion code for the benchmarking study reported in the paper:

> **ClusterDataSplit: Exploring Challenging Clustering-Based Data Splits for Model
Performance Evaluation** by Hanna Wecker, Annemarie Friedrich and Heike Adel.
In Proceedings of Evaluation and Comparison of NLP Systems (Eval4NLP).

The paper can be found [here](https://www.aclweb.org/anthology/2020.eval4nlp-1.15.pdf).
The code allows the users to reproduce and extend the results reported in the study.
You can also use the code to generate challenging data splits for your own datasets.
Please cite the above paper when reporting, reproducing or extending the results.

In case of questions, please contact the authors as listed on the paper.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of
the publication cited above. It will neither be maintained nor monitored in any way.

## Installing and Using ClusterDataSplit
ClusterDataSplit simply consists of a suite of three Jupyter notebooks for
* (1) Data Analysis;
* (2) Generation of Data Splits using a variety of algorithms including the Size
  and Distribution Sensitive K-Means algorithm; and
* (3) Model Performance Comparison.

In order to use ClusterDataSplit, install [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install/notebook-classic.html).
Install the Conda environment as described by `clusterdatasplit.yml`, see e.g., [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for how to do that.
Then navigate to the ClusterDataSplit top folder and start the Jupyter notebook server
by typing `jupyter notebook`. Your browser should open now and you can open and use
the notebooks.

**Note**: ClusterDataSplit only works with `sklearn (scikit-learn) <= 0.23.x`.

## License

ClusterDataSplit is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in ClusterDataSplit, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt). Our implementation of the Same-Size-K-Means algorithm follows the structure of that in the in the [ELKI Data Mining software](https://elki-project.github.io), but we have  re-implemented the algorithm in Python.

The sample data provided in the data folder are released unter the [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode.txt) as documented in the [data/LICENSE](data/LICENSE) file. In particular, the patent data (titles and abstracts) are sampled from the bulk data collection provided by [PatentsView](https://www.patentsview.org/download/).
