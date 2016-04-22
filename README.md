
Results
---------------

multi-label classification of the connectives.

```shell
Classifying 2945 test samples
             precision    recall  f1-score   support

          0       1.00      0.01      0.03        77
          1       0.66      0.15      0.25       216
          2       0.67      0.82      0.73       609
          3       0.00      0.00      0.00         1
          4       0.97      0.60      0.75       235
          5       1.00      0.65      0.79       133
          6       0.99      0.80      0.88       235
          7       0.00      0.00      0.00         6
          8       0.93      0.42      0.58        33
          9       0.90      0.82      0.86        22
         10       0.97      0.93      0.95       884
         11       0.00      0.00      0.00         3
         12       1.00      0.89      0.94        44
         13       1.00      0.20      0.33        25
         14       0.00      0.00      0.00         0
         15       0.00      0.00      0.00         1
         16       0.91      0.85      0.88       150
         17       0.99      0.51      0.67       155
         18       0.71      0.76      0.73       271

avg / total       0.86      0.73      0.76      3100

Accuracy: 0.725297113752

```



Ipython Notebooks
----------------
Contain features extracted from the training data provided.


Installation exploratory data analyisis
-----------
Clone the repository from github

```shell
git clone https://github.com/SBelkaid/NLPToolkits.git
``` 

Usage
-----

This is a script to explore the data, feel free to change anything you want.
Change the USE variable to one of the two variables above to change the path to the data you want to use. Two paths are given of which the first is the trial data provided by the tutorial and the second is the path to the actual train data. Make sure that you put the folder in the same relative path as specified in the script. 

```shell

NameOfComputer: python exploratory_conll.py

```


Shared Task Data Exploration
------------

distribution of relations in the training set.

```shell 
Counter({u'Comparison': 347,
         u'Comparison.Concession': 1080,
         u'Comparison.Contrast': 2956,
         u'Contingency': 1,
         u'Contingency.Cause.Reason': 943,
         u'Contingency.Cause.Result': 487,
         u'Contingency.Condition': 1148,
         u'Expansion': 24,
         u'Expansion.Alternative': 195,
         u'Expansion.Alternative.Chosen alternative': 96,
         u'Expansion.Conjunction': 4323,
         u'Expansion.Exception': 13,
         u'Expansion.Instantiation': 236,
         u'Expansion.Restatement': 121,
         u'Temporal': 4,
         u'Temporal.Asynchronous': 3,
         u'Temporal.Asynchronous.Precedence': 770,
         u'Temporal.Asynchronous.Succession': 842,
         u'Temporal.Synchrony': 1133})
```

![alt tag](https://raw.githubusercontent.com/SBelkaid/NLPToolkits/master/images/all_explicit.png)

The following is an image of the amount of unique cue phrases per relation sense:

![alt tag](https://raw.githubusercontent.com/SBelkaid/NLPToolkits/master/images/unique.png)

Output of comparing the salience of both arguments of one document using tfidf. Usually resulting in a higher salience for the first argument.

```shell
ARG1: Solo woodwind players have to be creative 

SALIENCE ARG1 1.96610017517 

CONECTIVE: IF 

SENSE: Contingency.Condition 

ARG2: they want to work a lot 

SALIENCE ARG2 1.72814871799 





ARG1: Solo woodwind players have to be creative if they want to work a lot 

SALIENCE ARG1 2.50204455491 

CONECTIVE: BECAUSE 

SENSE: Contingency.Cause.Reason 

ARG2: their repertoire and audience appeal are limited 

SALIENCE ARG2 1.92700701696 





ARG1: He commissions and splendidly interprets fearsome contemporary scores and does some conducting 

SALIENCE ARG1 2.76420720709 

CONECTIVE: SO 

SENSE: Contingency.Cause.Result 

ARG2: he doesn't have to play the same Mozart and Strauss concertos over and over again 

SALIENCE ARG2 2.12111288707 





ARG1: Today, the pixie-like clarinetist has mostly dropped the missionary work 

SALIENCE ARG1 2.40192146751 

CONECTIVE: THOUGH 

SENSE: Comparison.Concession 

ARG2: a touch of the old Tashi still survives 

SALIENCE ARG2 1.9027858474 





ARG1: Just the thing for the Vivaldi-at-brunch set, the yuppie audience that has embraced New Age as its very own easy listening 

SALIENCE ARG1 3.24105620405 

CONECTIVE: BUT 

SENSE: Comparison.Concession 

ARG2: you can't dismiss Mr. Stoltzman's music or his motives as merely commercial and lightweight 

SALIENCE ARG2 2.67478028335 





ARG1: He believes in what he plays 

SALIENCE ARG1 1.40134849657 

CONECTIVE: AND 

SENSE: Expansion.Conjunction 

ARG2: he plays superbly 

SALIENCE ARG2 1.39120620883 





ARG1: that his new album, "Inner Voices," had just been released, that his family was in the front row 

SALIENCE ARG1 2.65605945037 

CONECTIVE: AND 

SENSE: Expansion.Conjunction 

ARG2: that it was his mother's birthday 

SALIENCE ARG2 1.38858359134 





ARG1: and that it was his mother's birthday 

SALIENCE ARG1 1.38858359134 

CONECTIVE: SO 

SENSE: Contingency.Cause.Result 

ARG2: he was going to play her favorite tune from the record 

SALIENCE ARG2 2.15514823413 





ARG1: He launched into Saint-Saens's "The Swan" from "Carnival of the Animals," a favorite encore piece for cellists, with lovely, glossy tone and no bite 

SALIENCE ARG1 3.68486563828 

CONECTIVE: THEN 

SENSE: Temporal.Asynchronous.Precedence 

ARG2: he offered the second movement from Saint-Saens's Sonata for Clarinet, a whimsical, puckish tidbit that reflected the flip side of the Stoltzman personality 

SALIENCE ARG2 3.6259050031 





ARG1: Then he offered the second movement from Saint-Saens's Sonata for Clarinet, a whimsical, puckish tidbit that reflected the flip side of the Stoltzman personality 

SALIENCE ARG1 3.6259050031 

CONECTIVE: AS IF 

SENSE: Expansion 

ARG2: to show that he could play fast as well 

SALIENCE ARG2 1.41419501089 


```

All the unique discourse connectives:

```shell

defaultdict(set,
            {u'Comparison': {u'although',
              u'as though',
              u'but',
              u'even as',
              u'even if',
              u'even though',
              u'however',
              u'in fact',
              u'much as',
              u'nevertheless',
              u'nonetheless',
              u'still',
              u'then',
              u'though',
              u'while',
              u'yet'},
             u'Comparison.Concession': {u'although',
              u'and',
              u'as if',
              u'as much as',
              u'but',
              u'even after',
              u'even as',
              u'even if',
              u'even still',
              u'even then',
              u'even though',
              u'even when',
              u'however',
              u'if',
              u'in the end',
              u'meanwhile',
              u'nevertheless',
              u'nonetheless',
              u'regardless',
              u'still',
              u'though',
              u'when',
              u'while',
              u'yet'},
             u'Comparison.Contrast': {u'although',
              u'and',
              u'besides',
              u'but',
              u'by comparison',
              u'by contrast',
              u'conversely',
              u'earlier',
              u'even as',
              u'even though',
              u'however',
              u'if',
              u'if only',
              u'in contrast',
              u'in fact',
              u'in the end',
              u'instead',
              u'meanwhile',
              u'neither nor',
              u'nevertheless',
              u'nonetheless',
              u'nor',
              u'on the contrary',
              u'on the other hand',
              u'or',
              u'previously',
              u'rather',
              u'still',
              u'then',
              u'though',
              u'when',
              u'whereas',
              u'while',
              u'yet'},
             u'Contingency': {u'when'},
             u'Contingency.Cause.Reason': {u'apparently because',
              u'as',
              u'at least partly because',
              u'because',
              u'especially as',
              u'especially since',
              u'for',
              u'in large part because',
              u'in part because',
              u'indeed',
              u'insofar as',
              u'just because',
              u'largely because',
              u'mainly because',
              u'not because',
              u'not only because',
              u'now that',
              u'only because',
              u'particularly as',
              u'particularly because',
              u'particularly since',
              u'particularly when',
              u'partly because',
              u'perhaps because',
              u'primarily because',
              u'simply because',
              u'since',
              u'so',
              u'when'},
             u'Contingency.Cause.Result': {u'accordingly',
              u'and',
              u'as a result',
              u'but',
              u'consequently',
              u'hence',
              u'if only',
              u'in the end',
              u'in turn',
              u'largely as a result',
              u'now that',
              u'so',
              u'so that',
              u'then',
              u'thereby',
              u'therefore',
              u'thus'},
             u'Contingency.Condition': {u'and',
              u'as long as',
              u'at least when',
              u'especially if',
              u'especially when',
              u'even if',
              u'even when',
              u'if',
              u'if and when',
              u'if only',
              u'if then',
              u'just because',
              u'lest',
              u'once',
              u'only if',
              u'only when',
              u'particularly if',
              u'typically, if',
              u'unless',
              u'until',
              u'when'},
             u'Expansion': {u'and',
              u'as',
              u'as if',
              u'but',
              u'finally',
              u'in fact',
              u'in the end',
              u'indeed',
              u'or',
              u'ultimately'},
             u'Expansion.Alternative': {u'alternatively',
              u'as an alternative',
              u'besides',
              u'either or',
              u'else',
              u'except',
              u'except when',
              u'instead',
              u'just until',
              u'lest',
              u'neither nor',
              u'nor',
              u'or',
              u'otherwise',
              u'separately',
              u'then',
              u'unless',
              u'until',
              u'when'},
             u'Expansion.Alternative.Chosen alternative': {u'as much as',
              u'but',
              u'instead',
              u'rather',
              u'so much as'},
             u'Expansion.Conjunction': {u'additionally',
              u'also',
              u'and',
              u'as well',
              u'besides',
              u'but',
              u'even then',
              u'finally',
              u'further',
              u'furthermore',
              u'however',
              u'in addition',
              u'in fact',
              u'in the end',
              u'in the meantime',
              u'in turn',
              u'indeed',
              u'just as',
              u'later',
              u'likewise',
              u'meanwhile',
              u'moreover',
              u'neither nor',
              u'next',
              u'nor',
              u'on the other hand',
              u'or',
              u'overall',
              u'plus',
              u'separately',
              u'similarly',
              u'specifically',
              u'then',
              u'ultimately',
              u'while',
              u'yet'},
             u'Expansion.Exception': {u'although',
              u'but',
              u'except',
              u'otherwise'},
             u'Expansion.Instantiation': {u'and',
              u'for example',
              u'for instance',
              u'in fact',
              u'in particular',
              u'indeed'},
             u'Expansion.Restatement': {u'also',
              u'and',
              u'as though',
              u'but',
              u'for example',
              u'if',
              u'in fact',
              u'in other words',
              u'in particular',
              u'in short',
              u'in sum',
              u'in the end',
              u'in turn',
              u'indeed',
              u'much as',
              u'or',
              u'overall',
              u'rather',
              u'specifically',
              u'ultimately'},
             u'Temporal': {u'when'},
             u'Temporal.Asynchronous': {u'before and after',
              u'in the meantime',
              u'in turn'},
             u'Temporal.Asynchronous.Precedence': {u'a day or two before',
              u'a decade before',
              u'a full five minutes before',
              u'a week before',
              u'about six months before',
              u'afterward',
              u'afterwards',
              u'an average of six months before',
              u'and',
              u'at least until',
              u'before',
              u'but',
              u'by then',
              u'even before',
              u'ever since',
              u'finally',
              u'five minutes before',
              u'fully eight months before',
              u'in the 3 1/2 years before',
              u'in the end',
              u'in turn',
              u'just before',
              u'just days before',
              u'later',
              u'later on',
              u'long before',
              u'next',
              u'now that',
              u'only until',
              u'several months before',
              u'shortly afterward',
              u'shortly before',
              u'shortly thereafter',
              u'still',
              u'then',
              u'thereafter',
              u'till',
              u'two days before',
              u'two years before',
              u'ultimately',
              u'until',
              u'when',
              u'years before'},
             u'Temporal.Asynchronous.Succession': {u'18 months after',
              u'25 years after',
              u'29 years and 11 months to the day after',
              u'a day after',
              u'a few months after',
              u'a few weeks after',
              u'a month after',
              u'a week after',
              u'a year after',
              u'about a week after',
              u'about three weeks after',
              u'after',
              u'almost immediately after',
              u'as',
              u'as soon as',
              u'before',
              u'by then',
              u'earlier',
              u'eight months after',
              u'even after',
              u'ever since',
              u'five years after',
              u'four days after',
              u'immediately after',
              u'in the first 25 minutes after',
              u'in the meantime',
              u'just 15 days after',
              u'just a day after',
              u'just after',
              u'just five months after',
              u'just minutes after',
              u'just when',
              u'less than a month after',
              u'long after',
              u'minutes after',
              u'months after',
              u'more than a year after',
              u'nearly a year and a half after',
              u'nearly two months after',
              u'now that',
              u'once',
              u'one day after',
              u'only after',
              u'only three years after',
              u'only two weeks after',
              u'only when',
              u'previously',
              u'reportedly after',
              u'right after',
              u'seven years after',
              u'shortly after',
              u'since',
              u'since before',
              u'some time after',
              u'sometimes after',
              u'soon after',
              u'thereafter',
              u'three months after',
              u'two days after',
              u'two weeks after',
              u'until',
              u'when',
              u'when and if',
              u'within a year after',
              u'within minutes after',
              u'years after'},
             u'Temporal.Synchrony': {u'almost simultaneously',
              u'also',
              u'as',
              u'as long as',
              u'as soon as',
              u'at least not when',
              u'at least when',
              u'back when',
              u'by then',
              u'especially as',
              u'especially when',
              u'even as',
              u'even when',
              u'even while',
              u'in the meantime',
              u'in the meanwhile',
              u'just as',
              u'just as soon as',
              u'just when',
              u'meanwhile',
              u'now that',
              u'once',
              u'only as long as',
              u'only when',
              u'particularly as',
              u'simultaneously',
              u'since',
              u'then',
              u'until',
              u'when',
              u'while'}})
```



